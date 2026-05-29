"""
https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter/python?hl=zh-cn
https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter/index?hl=zh-cn#models
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
from tqdm import tqdm
from torch import nn
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torchvision.transforms.functional import pil_to_tensor


class MediaPipeSegmenter(object):
    """
    0 - background
    1 - hair
    2 - body-skin
    3 - face-skin
    4 - clothes
    5 - others (accessories)
    """

    def __init__(self, checkpoint_path="modules/mediapipe/selfie_multiclass_256x256.tflite"):
        """_summary_

        Args:
            checkpoint_path (str, optional): _description_. Defaults to "mediapipe/selfie_multiclass_256x256.tflite".
        """
        base_options = python.BaseOptions(model_asset_path=checkpoint_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options, running_mode=mp.tasks.vision.RunningMode.IMAGE, output_category_mask=True
        )
        self.model = vision.ImageSegmenter.create_from_options(options)

    @staticmethod
    def preprocessing(images):
        """_summary_

        Args:
            images (_type_): _description_

        Returns:
            _type_: _description_
        """
        images_resized = F.interpolate(images, (256, 256), mode="bilinear", align_corners=False)

        if images_resized.dtype != torch.uint8:
            images_resized = (255 * images_resized).to(torch.uint8)
        images_np = images_resized.permute(0, 2, 3, 1).detach().cpu().numpy()

        if not images_np.flags['C_CONTIGUOUS']:
            images_np = np.ascontiguousarray(images_np)
        images_np = images_np.astype(np.uint8)
        return images_np

    @staticmethod
    def postprocessing(images, H, W):
        """_summary_

        Args:
            images (_type_): _description_
            H (_type_): _description_
            W (_type_): _description_

        Returns:
            _type_: _description_
        """
        resized = np.zeros((images.shape[0], H, W), dtype=images.dtype)
        for i, image in enumerate(images):
            resized[i] = cv2.resize(image, (H, W), interpolation=cv2.INTER_NEAREST)
        return resized

    def __call__(self, images: torch.Tensor):
        """_summary_

        Args:
            images (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        H, W = images.shape[-2:]
        images_np = self.preprocessing(images)

        category_masks = []
        for image_np in tqdm(images_np, total=len(images_np), desc="Face Parser"):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
            segmented_masks = self.model.segment(mp_image)
            category_mask = segmented_masks.category_mask.numpy_view()
            category_masks.append(category_mask)
        category_masks = np.stack(category_masks, axis=0)

        category_masks = self.postprocessing(category_masks, H, W)

        return category_masks
