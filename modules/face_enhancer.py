import cv2
import numpy as np
import torch
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from typing import Literal


class FaceEnhancer(object):

    def __init__(
        self,
        model_name: Literal["esrgan", "gfpgan"] = "esrgan",
        realesrgan_path: str = "modules/face_enhancer/RealESRGAN_x4plus.pth",
        gfpgan_pat: str = "modules/face_enhancer/GFPGANv1.4.pth",
        outscale: Literal[1, 2, 3, 4] = 4,
        use_face_enhance: bool = True,
    ):
        """_summary_

        Args:
            model_name (Literal[&quot;esrgan&quot;, &quot;gfpgan&quot;], optional): _description_. Defaults to "esrgan".
            realesrgan_path (str, optional): _description_. Defaults to "modules/face_enhancer/RealESRGAN_x4plus.pth".
            gfpgan_pat (str, optional): _description_. Defaults to "modules/face_enhancer/GFPGANv1.4.pth".
            outscale (Literal[1, 2, 3, 4], optional): _description_. Defaults to 4.
            use_face_enhance (bool, optional): _description_. Defaults to True.

        Raises:
            NotImplementedError: _description_
        """
        if model_name == "esrgan":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.upsampler = RealESRGANer(
                scale=outscale,
                model_path=realesrgan_path,
                dni_weight=None,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True,
                gpu_id=0,
            )
            if use_face_enhance:
                self.face_enhancer = GFPGANer(
                    model_path=gfpgan_pat,
                    # https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
                    upscale=outscale,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=self.upsampler,
                )
        elif model_name == "gfpgan":
            self.restorer = GFPGANer(
                model_path=gfpgan_pat, upscale=outscale, arch='clean', channel_multiplier=2, bg_upsampler=None
            )
        else:
            raise NotImplementedError("`model_name` should be `esrgan` or `gfpgan`")

        self.model_name = model_name
        self.outscale = outscale
        self.use_face_enhance = use_face_enhance

    @staticmethod
    def tensor_to_opencv(tensor):
        """_summary_

        Args:
            tensor (_type_): _description_

        Returns:
            _type_: _description_
        """
        if len(tensor.shape) == 4:
            tensor = tensor[0]

        np_img = tensor.permute(1, 2, 0).cpu().numpy()

        if np_img.max() <= 1.0:
            np_img = (np_img * 255).astype(np.uint8)
        elif np_img.min() >= -1.0 and np_img.max() <= 1.0:
            np_img = ((np_img + 1) * 127.5).astype(np.uint8)
        else:
            np_img = np_img.astype(np.uint8)

        opencv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        return opencv_img

    @staticmethod
    def opencv_to_tensor(cv_image, normalize=True, device=None):
        """_summary_

        Args:
            cv_image (_type_): _description_
            normalize (bool, optional): _description_. Defaults to True.
            device (_type_, optional): _description_. Defaults to None.

        Raises:
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        if not isinstance(cv_image, np.ndarray):
            raise TypeError("The input must be of type numpy.ndarray (OpenCV image)")

        if len(cv_image.shape) == 2:
            cv_image = cv_image[:, :, np.newaxis]

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).to(torch.uint8)

        if normalize:
            tensor /= 255.0

        if device is not None:
            tensor = tensor.to(device)

        return tensor

    def __call__(self, images, out_size=256):
        """_summary_

        Args:
            images (_type_): _description_
            out_size (int, optional): _description_. Defaults to 256.

        Returns:
            _type_: _description_
        """
        device = images.device
        restored_imgs = []
        for image in tqdm(images, total=len(images), desc="Face Enhancer"):
            image = self.tensor_to_opencv(image)
            assert isinstance(image, np.ndarray) and (len(image.shape) == 3) and image.shape[-1] == 3

            img = cv2.resize(image, (out_size, out_size))

            if self.model_name == "esrgan":
                if self.use_face_enhance:
                    _, _, restored_img = self.face_enhancer.enhance(
                        img, has_aligned=False, only_center_face=False, paste_back=True, weight=1.0
                    )
                else:
                    restored_img, _ = self.upsampler.enhance(img, outscale=3.5)
            elif self.model_name == "gfpgan":
                _, _, restored_img = self.restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)

            restored_img = cv2.resize(restored_img, (out_size, out_size))
            restored_img = self.opencv_to_tensor(restored_img, normalize=False, device=device)
            restored_imgs.append(restored_img)
        restored_imgs = torch.stack(restored_imgs, dim=0)
        return restored_imgs
