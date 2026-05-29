import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from insightface.utils import face_align
from transformers import CLIPVisionModelWithProjection
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from insightface.app import FaceAnalysis
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from typing import Optional, Union, List


class SD15InpaintIPAdapterFaceIDPipeline(object):

    def __init__(
        self,
        det_size: int = 256,
        scale: float = 1.0,
        torch_dtype: torch.dtype = torch.float16,
        variant: str = "fp16",
        device: Union[str, torch.device] = "cuda",
        seed: int = 42,
        proxy_port: int = 7890,
    ):
        """_summary_

        Args:
            det_size (int, optional): _description_. Defaults to 256.
            scale (float, optional): _description_. Defaults to 1.0.
            torch_dtype (torch.dtype, optional): _description_. Defaults to torch.float16.
            variant (str, optional): _description_. Defaults to "fp16".
            device (Union[str, torch.device], optional): _description_. Defaults to "cuda".
            seed (int, optional): _description_. Defaults to 42.
            proxy_port (int, optional): _description_. Defaults to 7890.
        """
        if proxy_port is not None:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            # Enable agents
            os.environ['HTTP_PROXY'] = f"http://127.0.0.1:{proxy_port}"
            os.environ['HTTPS_PROXY'] = f"http://127.0.0.1:{proxy_port}"
            # requests.exceptions.SSLError: (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url
            # https://stackoverflow.com/questions/75110981/sslerror-httpsconnectionpoolhost-huggingface-co-port-443-max-retries-exce
            os.environ['CURL_CA_BUNDLE'] = ""

        # Load ArcFace
        # If FaceAnalysis runs only on the CPU, you can refer to this:
        # https://github.com/deepinsight/insightface/issues/2394#issuecomment-1929310317
        self.app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

        self.det_size = det_size
        self.torch_dtype = torch_dtype
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)

        # Load CLIP
        self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch_dtype
        ).to(device)

        # Load SD
        self.sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
            image_encoder=self.clip_image_encoder,
            torch_dtype=torch_dtype,
            variant=variant,
            safety_checker=None,  # Disable NSFW checking
            requires_safety_checker=False,  # Disable NSFW checking
        ).to(device)
        # self.sd_pipeline.enable_model_cpu_offload()
        self.sd_pipeline.set_progress_bar_config(disable=True)  # 禁用该 pipeline 自带的 tqdm 进度条

        # Load scheduler
        self.sd_pipeline.scheduler = DDIMScheduler.from_config(self.sd_pipeline.scheduler.config, eta=0.0)

        # Load Detail Tweaker LoRA
        self.sd_pipeline.load_lora_weights("OedoSoldier/detail-tweaker-lora")
        self.sd_pipeline.set_adapters(["default_0"], adapter_weights=[2.0])

        # Load IP-Adapter-FaceID
        self.sd_pipeline.load_ip_adapter(
            "h94/IP-Adapter-FaceID", subfolder=None, weight_name="ip-adapter-faceid-plus_sd15.bin", image_encoder_folder=None
        )
        self.sd_pipeline.set_ip_adapter_scale(scale)

    def prepare_id_embedding(self, id_images_list: List[Image.Image]):
        """_summary_

        Args:
            id_images_list (List[Image.Image]): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        id_embeddings = []
        last_valid_emb = None

        for i, id_image in enumerate(id_images_list):
            face_image_rgb = np.array(id_image)
            face_image_bgr = face_image_rgb[..., ::-1]
            faces = self.app.get(face_image_bgr)

            if not faces:
                if last_valid_emb is None:
                    raise ValueError(
                        f"No face was detected in frame {i}, and there are no available previous frames to fall back on!"
                    )
                print(
                    f"[Warning] ID feature extraction: No face was detected in frame {i}; features from the previous valid frame will be used automatically."
                )
                emb = last_valid_emb
            else:
                emb = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
                last_valid_emb = emb

            id_embeddings.append(emb)

        id_embedding_tensor = torch.stack(id_embeddings, dim=0)

        B = id_embedding_tensor.shape[0]
        smoothed_id_tensor = id_embedding_tensor.mean(dim=0, keepdim=True).repeat(B, 1, 1)
        # smoothed_id_tensor = id_embedding_tensor

        # Classifier-Free Guidance: (2B, 1, 512)
        neg_id_embeddings = torch.zeros_like(smoothed_id_tensor)
        final_id_embedding = torch.cat([neg_id_embeddings, smoothed_id_tensor]).to(dtype=self.torch_dtype, device=self.device)

        return final_id_embedding

    def prepare_structure_embedding(self, structure_images_list: List[Image.Image]):
        """_summary_

        Args:
            structure_images_list (List[Image.Image]): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        cond_embeds_list = []
        uncond_embeds_list = []
        last_valid_image = None

        for i, struct_image in enumerate(structure_images_list):
            struct_image_rgb = np.array(struct_image)
            struct_image_bgr = struct_image_rgb[..., ::-1]

            struct_faces = self.app.get(struct_image_bgr)

            if not struct_faces:
                if last_valid_image is None:
                    raise ValueError(
                        f"No face was detected in frame {i}, and there are no available previous frames to fall back on!"
                    )
                print(
                    f"[Warning] ID feature extraction: No face was detected in frame {i}; features from the previous valid frame will be used automatically."
                )
                face_image_pil = last_valid_image
            else:
                kps = struct_faces[0].kps
                face_image_arr, _ = face_align.norm_crop2(struct_image_rgb, landmark=kps, image_size=self.det_size)
                face_image_pil = Image.fromarray(face_image_arr)
                last_valid_image = face_image_pil

            cond_emb, uncond_emb = self.sd_pipeline.encode_image(
                face_image_pil, self.device, num_images_per_prompt=1, output_hidden_states=True
            )

            cond_embeds_list.append(cond_emb.to(dtype=self.torch_dtype))
            uncond_embeds_list.append(uncond_emb.to(dtype=self.torch_dtype))

        cond_embeds = torch.cat(cond_embeds_list, dim=0)
        uncond_embeds = torch.cat(uncond_embeds_list, dim=0)

        window_size = 1
        B = cond_embeds.shape[0]
        pad = window_size // 2
        smoothed_cond = torch.zeros_like(cond_embeds)
        smoothed_uncond = torch.zeros_like(uncond_embeds)

        for i in range(B):
            start_idx = max(0, i - pad)
            end_idx = min(B, i + pad + 1)

            smoothed_cond[i] = cond_embeds[start_idx:end_idx].mean(dim=0)
            smoothed_uncond[i] = uncond_embeds[start_idx:end_idx].mean(dim=0)

        # (B, 1, seq_len, dim)
        cond_embeds = smoothed_cond.unsqueeze(1)
        uncond_embeds = smoothed_uncond.unsqueeze(1)

        # (2B, 1, seq_len, dim)
        final_clip_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)

        return final_clip_embeds.detach()

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

        if prompt is None:
            prompt = ""
        if negative_prompt is None:
            # negative_prompt = "blurry, distorted lips, extra teeth, unrealistic mouth, artifacts"
            negative_prompt = "missing crooked deformed teeth, distorted lips, unrealistic mouth, blurry, artifacts"

        id_clips = list(torch.split(id_video, split_size_or_sections=clip_len, dim=0))
        driven_clips = list(torch.split(driven_video, split_size_or_sections=clip_len, dim=0))
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

        output_clips = []
        for id_clip, driven_clip, mask_clip in tqdm(zip(id_clips, driven_clips, mask_clips), total=len(id_clips)):
            id_clip = [to_pil_image(frame) for frame in id_clip]
            driven_clip = [to_pil_image(frame) for frame in driven_clip]
            mask_clip = [to_pil_image(frame) for frame in mask_clip]

            init_noise_clip = init_noise.repeat(len(id_clip), 1, 1, 1)

            prompts = [prompt] * len(id_clip)
            negative_prompts = [negative_prompt] * len(id_clip)

            id_embedding = self.prepare_id_embedding(id_clip)
            clip_embedding = self.prepare_structure_embedding(driven_clip)

            self.sd_pipeline.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = clip_embedding.to(
                dtype=self.torch_dtype
            )
            self.sd_pipeline.unet.encoder_hid_proj.image_projection_layers[0].shortcut = False

            original_prepare_ip_adapter = self.sd_pipeline.prepare_ip_adapter_image_embeds
            self.sd_pipeline.prepare_ip_adapter_image_embeds = lambda *args, **kwargs: [id_embedding]

            output_images = self.sd_pipeline(
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

            output_images = [pil_to_tensor(img) for img in output_images]
            output_images = torch.stack(output_images, dim=0)
            output_clips.append(output_images)

        output_clips = torch.concat(output_clips, dim=0)
        return output_clips
