import cv2
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
from insightface.app import FaceAnalysis
from torchvision.transforms.functional import to_pil_image


class StructureController(object):

    def __init__(self, app: FaceAnalysis):
        self.app = app

    @staticmethod
    def smooth_heights(data: list[float], window_size: int) -> list[float]:
        """_summary_

        Args:
            data (list[float]): _description_
            window_size (int): _description_

        Returns:
            list[float]: _description_
        """
        data_np = np.array(data, dtype=np.float32)

        pad_len = window_size // 2

        padded_data = np.pad(data_np, pad_width=pad_len, mode="edge")

        window_sum = np.convolve(padded_data, np.ones(window_size), mode="valid")
        smoothed = window_sum / window_size

        return smoothed

    def get_mouth_heights(self, frames):
        """_summary_

        Args:
            frames (_type_): _description_

        Returns:
            _type_: _description_
        """
        heights = []
        for frame in tqdm(frames, total=len(frames), desc="Structure Controller [Lambda]"):
            frame_cv = cv2.cvtColor(np.asarray(to_pil_image(frame)), cv2.COLOR_RGB2BGR)
            face = self.app.get(frame_cv)[0]
            kps = face.landmark_2d_106
            left_top = kps[66]
            left_bottom = kps[54]
            top = kps[62]
            bottom = kps[60]
            right_top = kps[70]
            right_bottom = kps[57]
            height = (
                np.linalg.norm(left_top - left_bottom) + np.linalg.norm(top - bottom) + np.linalg.norm(right_top - right_bottom)
            ) / 3
            heights.append(height)
        heights = self.smooth_heights(heights, window_size=5)
        return heights

    def get_lambdas(self, frames):
        """_summary_

        Args:
            frames (_type_): _description_

        Returns:
            _type_: _description_
        """
        heights = self.get_mouth_heights(frames)
        lambdas = heights[1:] / heights[:-1]
        lambdas = np.insert(lambdas, 0, 1.0)
        lambdas = torch.from_numpy(lambdas)[..., None, None]
        return lambdas

    def __call__(self, anchor_embeds, clip_embeds, lambdas):
        """_summary_

        Args:
            anchor_embeds (_type_): _description_
            clip_embeds (_type_): _description_
            lambdas (_type_): _description_

        Returns:
            _type_: _description_
        """
        clip_embeds = list(torch.split(clip_embeds, split_size_or_sections=2, dim=0))  # list[(2, 1, 257, 1280)]
        clip_embeds = torch.stack(clip_embeds, dim=0)  # (T, 2, 1, 257, 1280)
        e_start = anchor_embeds[:, 1, 0, 1:]  # (1, 257, 1280)
        e_end = clip_embeds[:, 1, 0, 1:]  # (T, 257, 1280)
        e_end_new = e_start + lambdas * (e_end - e_start)
        clip_embeds[:, 1, 0, 1:] = e_end_new.detach()  # (T, 2, 1, 257, 1280)
        clip_embeds = rearrange(clip_embeds, "t a b c d -> (t a) b c d")
        return clip_embeds
