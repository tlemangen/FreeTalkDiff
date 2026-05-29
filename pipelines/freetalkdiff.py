import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from typing import Optional, Union

from pipelines import SD15InpaintIPAdapterFaceIDPipeline
from modules import Masker, MediapipeStructurist, StructureController, NoiseSensor


class FreeTalkDiff(SD15InpaintIPAdapterFaceIDPipeline):

    def __init__(
        self,
        cfg: DictConfig,
        det_size: int = 256,
        scale: float = 1.0,
        torch_dtype: torch.dtype = torch.float16,
        variant: str = "fp16",
        device: Union[str, torch.device] = "cuda",
        seed: int = 42,
        proxy_port: int = 7892,
    ):
        """_summary_

        Args:
            cfg (DictConfig): _description_
            det_size (int, optional): _description_. Defaults to 256.
            scale (float, optional): _description_. Defaults to 1.0.
            torch_dtype (torch.dtype, optional): _description_. Defaults to torch.float16.
            variant (str, optional): _description_. Defaults to "fp16".
            device (Union[str, torch.device], optional): _description_. Defaults to "cuda".
            seed (int, optional): _description_. Defaults to 42.
            proxy_port (int, optional): _description_. Defaults to 7892.
        """
        super().__init__(det_size, scale, torch_dtype, variant, device, seed, proxy_port)
        self.masker = Masker(**cfg.masker)
        self.structurist = MediapipeStructurist(**cfg.structurist)
        self.structure_controller = StructureController(self.app)
        self.noise_sensor = NoiseSensor()

    @torch.inference_mode()
    def __call__(
        self,
        id_video: torch.Tensor,
        driven_video: torch.Tensor,
        mask_video: Optional[torch.Tensor] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        strength: float = 1.0,
        clip_len: int = 5,
    ):
        """_summary_

        Args:
            id_video (torch.Tensor): _description_
            driven_video (torch.Tensor): _description_
            mask_video (Optional[torch.Tensor], optional): _description_. Defaults to None.
            prompt (Optional[str], optional): _description_. Defaults to None.
            negative_prompt (Optional[str], optional): _description_. Defaults to None.
            guidance_scale (float, optional): _description_. Defaults to 7.5.
            num_inference_steps (int, optional): _description_. Defaults to 50.
            strength (float, optional): _description_. Defaults to 1.0.
            clip_len (int, optional): _description_. Defaults to 5.

        Returns:
            _type_: _description_
        """
        assert id_video.shape == driven_video.shape, "`id_videop` and `driven_video` should have the same shape."

        T, C, H, W = id_video.shape

        if mask_video is None:
            mask_video = torch.zeros((T, 1, H, W), dtype=torch.uint8)
            mask_video[..., H // 2 :, W // 4 : -W // 4] = 255
            # mask_video = self.masker(id_video, mask_body=True, mask_face=True, do_blur=False)

        if prompt is None:
            prompt = ""
        if negative_prompt is None:
            # negative_prompt = "blurry, distorted lips, extra teeth, unrealistic mouth, artifacts"
            negative_prompt = "missing crooked deformed teeth, distorted lips, unrealistic mouth, blurry, artifacts"

        structure_video = self.structurist(id_video, driven_video)

        id_clips = list(torch.split(id_video, split_size_or_sections=clip_len, dim=0))
        structure_clips = list(torch.split(structure_video, split_size_or_sections=clip_len, dim=0))
        mask_clips = list(torch.split(mask_video, split_size_or_sections=clip_len, dim=0))

        init_noise = torch.randn(
            (
                1,
                self.sd_pipeline.vae.config.latent_channels,
                H // self.sd_pipeline.vae_scale_factor,
                W // self.sd_pipeline.vae_scale_factor,
            ),
            device=self.device,
            dtype=self.torch_dtype,
            generator=self.generator,
        )

        # Initialize Structure Controller
        lambdas = self.structure_controller.get_lambdas(driven_video).to(self.device)
        lambda_clips = list(torch.split(lambdas, split_size_or_sections=clip_len, dim=0))
        anchor_embedding = self.prepare_structure_embedding([to_pil_image(structure_video[0])]).unsqueeze(0)

        output_clips = []
        for id_clip, structure_clip, mask_clip, lambda_clip in tqdm(
            zip(id_clips, structure_clips, mask_clips, lambda_clips), total=len(id_clips), desc="Stable Diffusion"
        ):
            id_clip = [to_pil_image(frame) for frame in id_clip]
            structure_clip = [to_pil_image(frame) for frame in structure_clip]
            mask_clip = [to_pil_image(frame) for frame in mask_clip]

            init_noise_clip = init_noise.repeat(len(id_clip), 1, 1, 1)

            prompts = [prompt] * len(id_clip)
            negative_prompts = [negative_prompt] * len(id_clip)

            id_embedding = self.prepare_id_embedding(id_clip)
            clip_embedding = self.prepare_structure_embedding(structure_clip)

            # Structure Controller
            structure_embedding = self.structure_controller(anchor_embedding, clip_embedding, lambda_clip)

            self.sd_pipeline.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = structure_embedding.to(
                dtype=self.torch_dtype
            )
            self.sd_pipeline.unet.encoder_hid_proj.image_projection_layers[0].shortcut = False

            original_prepare_ip_adapter = self.sd_pipeline.prepare_ip_adapter_image_embeds
            self.sd_pipeline.prepare_ip_adapter_image_embeds = lambda *args, **kwargs: [id_embedding]

            output_clip = self.sd_pipeline(
                latents=init_noise_clip,
                height=H,
                width=W,
                prompt=prompts,
                negative_prompt=negative_prompts,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=self.generator,
                image=id_clip,
                mask_image=mask_clip,
                ip_adapter_image_embeds=[id_embedding],
                strength=strength,
                output_type="pil",
            ).images

            self.sd_pipeline.prepare_ip_adapter_image_embeds = original_prepare_ip_adapter

            output_clip = [pil_to_tensor(frame) for frame in output_clip]
            output_clip = torch.stack(output_clip, dim=0)
            output_clips.append(output_clip)

        output_video = torch.concat(output_clips, dim=0)
        output_video = self.noise_sensor(output_video, id_video).to(torch.uint8)
        return output_video
