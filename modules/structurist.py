import cv2
import numpy as np
import torch
import mediapipe as mp
from tqdm import tqdm
from pathlib import Path


class MediapipeStructurist(object):
    INNER_LIP_LANDMARKS = np.asarray(
        [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95], dtype=np.int32
    )

    def __init__(
        self,
        canonical_face_model_path: str = "modules/mediapipe/canonical_face_model.obj",
        texture_size: int = 256,
        static_image_mode: bool = False,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        preserve_mouth_hole: bool = True,
    ):
        """_summary_

        Args:
            canonical_face_model_path (str, optional): _description_. Defaults to "modules/mediapipe/canonical_face_model.obj".
            texture_size (int, optional): _description_. Defaults to 256.
            static_image_mode (bool, optional): _description_. Defaults to False.
            refine_landmarks (bool, optional): _description_. Defaults to True.
            min_detection_confidence (float, optional): _description_. Defaults to 0.5.
            min_tracking_confidence (float, optional): _description_. Defaults to 0.5.
            preserve_mouth_hole (bool, optional): _description_. Defaults to True.
        """
        self.texture_size = texture_size
        self.static_image_mode = static_image_mode
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.preserve_mouth_hole = preserve_mouth_hole
        self.uv, self.faces = self.load_canonical_face_model(canonical_face_model_path)

    def load_canonical_face_model(self, canonical_face_model_path):
        """Load MediaPipe canonical mesh UVs and triangles."""
        candidate_paths = [
            Path(canonical_face_model_path),
            Path(mp.__file__).resolve().parent / "modules" / "face_geometry" / "data" / "canonical_face_model.obj",
        ]
        canonical_obj = next((path for path in candidate_paths if path.exists()), None)
        if canonical_obj is None:
            attempted = "\n".join(f"- {path}" for path in candidate_paths)
            raise FileNotFoundError(
                "MediaPipe canonical face model not found. Tried:\n"
                f"{attempted}\n"
                "Place canonical_face_model.obj at modules/mediapipe/canonical_face_model.obj "
                "or install a MediaPipe build that includes face_geometry data."
            )

        raw_uvs = []
        face_vertices = []
        face_uvs = []
        max_vertex_idx = -1

        with canonical_obj.open("r", encoding="utf-8") as obj_file:
            for line in obj_file:
                line = line.strip()
                if line.startswith("vt "):
                    _, u, v = line.split()[:3]
                    raw_uvs.append([float(u), float(v)])
                elif line.startswith("f "):
                    vertices = []
                    uvs = []
                    for token in line.split()[1:4]:
                        parts = token.split("/")
                        vertex_idx = int(parts[0]) - 1
                        uv_idx = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else vertex_idx
                        vertices.append(vertex_idx)
                        uvs.append(uv_idx)
                        max_vertex_idx = max(max_vertex_idx, vertex_idx)
                    face_vertices.append(vertices)
                    face_uvs.append(uvs)

        raw_uvs = np.asarray(raw_uvs, dtype=np.float32)
        faces = np.asarray(face_vertices, dtype=np.int32)
        face_uvs = np.asarray(face_uvs, dtype=np.int32)

        uv_by_vertex = np.zeros((max_vertex_idx + 1, 2), dtype=np.float32)
        uv_seen = np.zeros((max_vertex_idx + 1,), dtype=bool)
        for vertices, uvs in zip(faces, face_uvs):
            for vertex_idx, uv_idx in zip(vertices, uvs):
                uv_by_vertex[vertex_idx] = raw_uvs[uv_idx]
                uv_seen[vertex_idx] = True

        if not uv_seen.all():
            missing = np.where(~uv_seen)[0][:10].tolist()
            raise ValueError(f"Canonical face model has vertices without UV coordinates: {missing}")

        return uv_by_vertex, faces

    @staticmethod
    def detect_face_landmarks(face_mesh, frame_rgb):
        """Return MediaPipe Face Mesh landmarks as image-space xy coordinates."""
        result = face_mesh.process(np.ascontiguousarray(frame_rgb))
        if not result.multi_face_landmarks:
            raise ValueError("MediaPipe Face Mesh did not detect a face in the frame")

        height, width = frame_rgb.shape[:2]
        landmarks = result.multi_face_landmarks[0].landmark
        xy = np.asarray([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float32)
        return xy

    @staticmethod
    def _uv_to_pixels(uv, tex_size):
        uv_px = uv.copy().astype(np.float32)
        uv_px[:, 0] *= tex_size - 1
        uv_px[:, 1] = (1.0 - uv_px[:, 1]) * (tex_size - 1)
        return uv_px

    @staticmethod
    def _warp_triangle(src, dst, src_tri, dst_tri):
        src_rect = cv2.boundingRect(src_tri.astype(np.float32))
        dst_rect = cv2.boundingRect(dst_tri.astype(np.float32))

        if src_rect[2] <= 0 or src_rect[3] <= 0 or dst_rect[2] <= 0 or dst_rect[3] <= 0:
            return

        src_offset = src_tri - np.asarray(src_rect[:2], dtype=np.float32)
        dst_offset = dst_tri - np.asarray(dst_rect[:2], dtype=np.float32)

        src_patch = src[src_rect[1] : src_rect[1] + src_rect[3], src_rect[0] : src_rect[0] + src_rect[2]]
        if src_patch.size == 0:
            return

        transform = cv2.getAffineTransform(src_offset.astype(np.float32), dst_offset.astype(np.float32))
        warped = cv2.warpAffine(
            src_patch, transform, (dst_rect[2], dst_rect[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )

        mask = np.zeros((dst_rect[3], dst_rect[2]), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.round(dst_offset).astype(np.int32), 1.0, lineType=cv2.LINE_AA)
        mask = mask[..., None]

        dst_roi = dst[dst_rect[1] : dst_rect[1] + dst_rect[3], dst_rect[0] : dst_rect[0] + dst_rect[2]]
        if dst_roi.shape[:2] != mask.shape[:2]:
            return
        dst_roi[:] = dst_roi * (1.0 - mask) + warped * mask

    def extract_uv_texture(self, frame_rgb, landmarks):
        """Bake visible source face pixels into the canonical MediaPipe UV atlas."""
        frame = frame_rgb.astype(np.float32)
        texture = np.zeros((self.texture_size, self.texture_size, 3), dtype=np.float32)
        uv_px = self._uv_to_pixels(self.uv, self.texture_size)

        for face in self.faces:
            src_tri = landmarks[face]
            dst_tri = uv_px[face]
            self._warp_triangle(frame, texture, src_tri, dst_tri)

        return np.clip(texture, 0, 255).astype(np.uint8)

    def _build_mouth_hole_mask(self, frame_shape, landmarks):
        mask = np.zeros(frame_shape[:2], dtype=np.float32)
        mouth_polygon = landmarks[self.INNER_LIP_LANDMARKS]
        cv2.fillPoly(mask, [np.round(mouth_polygon).astype(np.int32)], 1.0, lineType=cv2.LINE_AA)
        return mask[..., None]

    def _restore_mouth_hole(self, rendered, target_frame_rgb, target_landmarks):
        mask = self._build_mouth_hole_mask(rendered.shape, target_landmarks)
        return rendered * (1.0 - mask) + target_frame_rgb.astype(np.float32) * mask

    def render_uv_to_face(self, texture, target_frame_rgb, target_landmarks):
        """Paste a canonical MediaPipe UV atlas onto a target Face Mesh."""
        output = target_frame_rgb.astype(np.float32).copy()
        uv_px = self._uv_to_pixels(self.uv, texture.shape[0])

        for face in self.faces:
            src_tri = uv_px[face]
            dst_tri = target_landmarks[face]
            self._warp_triangle(texture.astype(np.float32), output, src_tri, dst_tri)

        if self.preserve_mouth_hole:
            output = self._restore_mouth_hole(output, target_frame_rgb, target_landmarks)

        return np.clip(output, 0, 255).astype(np.uint8)

    @staticmethod
    def _video_to_numpy_rgb(video):
        if video.amax() <= 1.0:
            video = (255 * video).to(torch.uint8)
        else:
            video = video.to(torch.uint8)
        return video.permute(0, 2, 3, 1).detach().cpu().numpy()

    @torch.inference_mode()
    def __call__(self, id_video, driven_video, return_comparison=False):
        """_summary_

        Args:
            id_video (_type_): _description_
            driven_video (_type_): _description_
            return_comparison (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        id_frames = self._video_to_numpy_rgb(id_video)
        driven_frames = self._video_to_numpy_rgb(driven_video)
        num_frames = min(len(id_frames), len(driven_frames))
        id_frames = id_frames[:num_frames]
        driven_frames = driven_frames[:num_frames]

        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=1,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as face_mesh:
            source_landmarks = self.detect_face_landmarks(face_mesh, id_frames[0])
            source_texture = self.extract_uv_texture(id_frames[0], source_landmarks)

            structure_images = []
            for i in tqdm(range(num_frames), total=num_frames, desc="MediaPipe Structurist"):
                target_landmarks = self.detect_face_landmarks(face_mesh, driven_frames[i])
                transferred = self.render_uv_to_face(source_texture, driven_frames[i], target_landmarks)

                out_img = torch.from_numpy(transferred).permute(2, 0, 1).to(id_video.device, torch.uint8)

                if return_comparison:
                    source = torch.from_numpy(id_frames[i]).permute(2, 0, 1).to(id_video.device, torch.uint8)
                    target = torch.from_numpy(driven_frames[i]).permute(2, 0, 1).to(id_video.device, torch.uint8)
                    out_img = torch.cat([source, out_img, target], dim=-1)

                structure_images.append(out_img)

        return torch.stack(structure_images, dim=0).detach().cpu()
