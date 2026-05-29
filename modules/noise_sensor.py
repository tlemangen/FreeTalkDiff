import cv2
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


class NoiseSensor(object):

    def __init__(self):
        pass

    @staticmethod
    def compute_covariance(x: torch.Tensor, y: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_
            unbiased (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            torch.Tensor: _description_
        """
        # 输入验证
        if x.shape != y.shape:
            raise ValueError(f"The shapes of x and y must be the same, but in reality, x: {x.shape}, y: {y.shape}")
        if x.ndim != 4:
            raise ValueError(f"The input tensor must be a 4D tensor with shape (T, H, W, C); the actual shape: {x.shape}")

        T = x.shape[0]
        if T < 2:
            raise ValueError(f"The time dimension T must be at least 2 to calculate the covariance; the actual T: {T}")

        mean_x = torch.mean(x, dim=0)  # (H, W, 2)
        mean_y = torch.mean(y, dim=0)  # (H, W, 2)

        x_dev = x - mean_x
        y_dev = y - mean_y

        cov_sum = torch.sum(x_dev * y_dev, dim=0)

        denom = T - 1 if unbiased else T

        covariance = cov_sum / denom
        return covariance

    @staticmethod
    def get_gaussian_kernel1d_batch(sigmas, kernel_size):
        """_summary_

        Args:
            sigmas (_type_): _description_
            kernel_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        H, W = sigmas.shape
        device = sigmas.device

        k = torch.arange(kernel_size, device=device).float() - (kernel_size - 1) / 2
        k = k.view(1, 1, -1).expand(H, W, -1)
        sigmas = sigmas.unsqueeze(-1)  # (H, W, 1)

        kernel = torch.exp(-(k**2) / (2 * (sigmas) ** 2 + 1e-6))
        return kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-6)

    def spatio_temporal_blur_adaptive(self, video, sigma_t, k_t=5, k_s=1, sigma_s=1.0):
        """_summary_

        Args:
            video (_type_): _description_
            sigma_t (_type_): _description_
            k_t (int, optional): _description_. Defaults to 5.
            k_s (int, optional): _description_. Defaults to 1.
            sigma_s (float, optional): _description_. Defaults to 1.0.

        Returns:
            _type_: _description_
        """
        T, C, H, W = video.shape
        device = video.device

        if k_s > 1:
            k_range = torch.arange(k_s, device=device).float() - (k_s - 1) / 2
            k_2d = torch.exp(-(k_range**2) / (2 * sigma_s**2))
            k_2d = k_2d[:, None] * k_2d[None, :]
            k_2d = (k_2d / k_2d.sum()).view(1, 1, k_s, k_s).repeat(C, 1, 1, 1)

            video = F.conv2d(video, k_2d, padding=k_s // 2, groups=C)

        kernels = self.get_gaussian_kernel1d_batch(sigma_t, k_t)

        video_tmp = video.permute(1, 2, 3, 0)
        video_tmp = torch.concat(
            [video_tmp[:, :, :, -(k_t // 2) :], video_tmp, video_tmp[:, :, :, : k_t // 2]], dim=-1
        )  # (C, H, W, T + k_t - 1)

        windows = video_tmp.unfold(-1, k_t, 1)

        smoothed_video = (windows * kernels.view(1, H, W, 1, k_t)).sum(dim=-1)

        return smoothed_video.permute(3, 0, 1, 2)

    def __call__(self, fake_video, real_video):
        """_summary_

        Args:
            fake_video (_type_): _description_
            real_video (_type_): _description_

        Returns:
            _type_: _description_
        """
        assert fake_video.shape == real_video.shape, "`fake_video` and `real_video` must have the same shape."
        num_frames = fake_video.shape[0]

        flows_fake = []
        flows_real = []
        for i in tqdm(range(num_frames - 1), total=num_frames - 1, desc="Noise Sensor"):
            gray_fake_prev = cv2.cvtColor(fake_video[i].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
            gray_fake_curr = cv2.cvtColor(fake_video[i + 1].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
            flow_fake = cv2.calcOpticalFlowFarneback(
                gray_fake_prev,
                gray_fake_curr,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )  # (height, width, 2)
            gray_real_prev = cv2.cvtColor(real_video[i].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
            gray_real_curr = cv2.cvtColor(real_video[i + 1].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
            flow_real = cv2.calcOpticalFlowFarneback(
                gray_real_prev,
                gray_real_curr,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )  # (height, width, 2)
            flows_fake.append(flow_fake)
            flows_real.append(flow_real)
        flows_fake = torch.from_numpy(np.stack(flows_fake, axis=0))  # (T, H, W, 2)
        flows_real = torch.from_numpy(np.stack(flows_real, axis=0))  # (T, H, W, 2)
        cov = self.compute_covariance(flows_real, flows_fake, unbiased=True)  # (H, W, 2)
        var_fake = torch.var(flows_fake, dim=0, unbiased=True)  # (H, W, 2)
        var_real = torch.var(flows_real, dim=0, unbiased=True)  # (H, W, 2)
        var_noise = var_fake - torch.square(cov) / (var_real + 1e-8)  # (H, W, 2)
        noise_pattern = torch.linalg.norm(var_noise, dim=-1)  # (H, W)
        output_video = self.spatio_temporal_blur_adaptive(
            fake_video.float(), sigma_t=noise_pattern, k_t=3
        ).int()  # (T, C, H, W), [0, 255]
        return output_video
