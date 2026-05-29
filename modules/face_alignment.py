import cv2
import torch
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from typing import Optional

from modules.masker import Masker


class FaceAlignment(object):

    def __init__(self, app: FaceAnalysis, masker: Masker, det_size: int = 256):
        """_summary_

        Args:
            app (FaceAnalysis): _description_
            masker (Masker): _description_
            det_size (int, optional): _description_. Defaults to 256.
        """
        self.app = app
        self.masker = masker
        self.image_size = det_size

    def align_and_crop(self, frames):
        face_images = []
        Ms = []
        for frame in tqdm(frames, total=len(frames), desc="Face Alignment [Align&Crop]"):
            frame = frame.permute(1, 2, 0)
            struct_image_rgb = np.array(frame)  # (H, W, C), [0, 255]
            struct_image_bgr = struct_image_rgb[..., ::-1]
            struct_faces = self.app.get(struct_image_bgr)
            kps = struct_faces[0].kps
            face_image, M = face_align.norm_crop2(struct_image_rgb, landmark=kps, image_size=self.image_size)
            face_images.append(torch.from_numpy(face_image))
            Ms.append(M)
        face_images = torch.stack(face_images).permute(0, 3, 1, 2)
        face_images = face_images.to(device=frames.device)
        Ms = np.stack(Ms)
        return face_images, Ms

    @staticmethod
    def inverse_affine_transform(cropped_img, M, original_size):
        """_summary_

        Args:
            cropped_img (_type_): _description_
            M (_type_): _description_
            original_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        if isinstance(cropped_img, torch.Tensor):
            cropped_img = cropped_img.detach().cpu().numpy()
            if cropped_img.shape[0] in [3, 1]:
                cropped_img = cropped_img.transpose(1, 2, 0)

        M_homogeneous = np.vstack([M, [0, 0, 1]])
        M_inv_homogeneous = np.linalg.inv(M_homogeneous)
        M_inv = M_inv_homogeneous[:2, :]

        h, w = original_size
        recovered_img = cv2.warpAffine(
            cropped_img, M_inv, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        return recovered_img

    def recover(self, cropped_frames, Ms=None, bg_frames=None):
        """_summary_

        Args:
            cropped_frames (_type_): _description_
            Ms (_type_, optional): _description_. Defaults to None.
            bg_frames (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if bg_frames is None:
            bg_frames = torch.zeros_like(cropped_frames)
        original_size = bg_frames.shape[-2:]

        mouth_masks = self.masker(cropped_frames, mask_body=False, mask_face=True, do_blur=True).repeat(1, 3, 1, 1)

        if Ms is not None:
            face_images = []
            masks = []
            for frame, M, mouth_mask in tqdm(
                zip(cropped_frames, Ms, mouth_masks), total=len(cropped_frames), desc="Face Alignment [Recover]"
            ):
                frame = frame.permute(1, 2, 0)
                mouth_mask = mouth_mask.permute(1, 2, 0)

                face_image = self.inverse_affine_transform(frame, M, original_size)
                mask = self.inverse_affine_transform(mouth_mask, M, original_size)

                face_images.append(torch.from_numpy(face_image))
                masks.append(torch.from_numpy(mask))
            cropped_frames = torch.stack(face_images).permute(0, 3, 1, 2)
            mouth_masks = torch.stack(masks).permute(0, 3, 1, 2)

        mouth_masks = mouth_masks / 255.0
        merged = (mouth_masks * cropped_frames + (1 - mouth_masks) * bg_frames).to(torch.uint8)

        return merged
