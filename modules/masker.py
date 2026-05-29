import cv2
import torch
import numpy as np

from modules.mediapipe_segmenter import MediaPipeSegmenter


class Masker(object):

    def __init__(self, checkpoint_path: str = "modules/mediapipe/selfie_multiclass_256x256.tflite"):
        self.face_parser = MediaPipeSegmenter(checkpoint_path)

    def __call__(self, frames, mask_body=True, mask_face=True, do_blur=False):
        """_summary_

        Args:
            frames (_type_): _description_
            mask_body (bool, optional): _description_. Defaults to True.
            mask_face (bool, optional): _description_. Defaults to True.
            do_blur (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        category_masks = self.face_parser(frames)  # (T, H, W)

        parse_mask = np.zeros(category_masks.shape, dtype=np.uint8)
        MASK_COLORMAP = [0, 0, int(float(mask_body) * 255), int(float(mask_face) * 255), 0, 0]
        for idx, color in enumerate(MASK_COLORMAP):
            parse_mask[category_masks == idx] = color

        mouth_mask = np.zeros(category_masks.shape, dtype=np.uint8)
        mouth_mask[..., mouth_mask.shape[-2] // 2 :, mouth_mask.shape[-1] // 4 : -(mouth_mask.shape[-1] // 4)] = 255
        mouth_mask[
            ...,
            mouth_mask.shape[-2] // 2 : 5 * mouth_mask.shape[-2] // 8,
            3 * mouth_mask.shape[-1] // 8 : -(3 * mouth_mask.shape[-1] // 8),
        ] = 0

        mouth_masks = ((mouth_mask / 255) * parse_mask).astype(np.uint8)

        if do_blur:
            #  blur the mask
            mouth_masks_blur = []
            for m in mouth_masks:
                m_blur = cv2.GaussianBlur(m, (25, 25), sigmaX=11, sigmaY=11)
                mouth_masks_blur.append(m_blur)
            mouth_masks = np.stack(mouth_masks_blur, axis=0)

        mouth_masks = torch.from_numpy(mouth_masks).unsqueeze(1).to(torch.uint8)  # (T, 1, H, W)
        return mouth_masks
