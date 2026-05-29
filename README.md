# FreeTalkDiff

---

[CVPR 2026] IP-Adapter Is All You Need: Towards Fine-Tuning-Free Diffusion-Based Talking Face Generation

[Hao Wu](https://github.com/tlemangen), Xiangyang Luo, Hao Wang, Jiawei Zhang, Yi Zhang, and Jinwei Wang

![GitHub License](https://img.shields.io/github/license/tlemangen/FreeTalkDiff) &nbsp; <a href="https://arxiv.org/abs/2605.30230"><img src="https://img.shields.io/badge/arXiv-PDF-red"></a> &nbsp; ![CVPR PDF](https://img.shields.io/badge/CVPR-PDF-blue) &nbsp; <a href="https://cvpr.thecvf.com/virtual/2026/poster/39232"><img src="https://img.shields.io/badge/CVPR-Poster-blue"></a> &nbsp; <a href="https://github.com/tlemangen/FreeTalkDiff"><img src="https://img.shields.io/badge/Github-100000.svg?logo=github&logoColor=white"></a>

<p align="center" width="100%">
<video src="
https://github.com/user-attachments/assets/a8fdd3aa-e4d4-4e63-add0-9f2118795765
" width="256" height="256" controls></video>
</p>

**Abstract**: With the rapid advancement of diffusion models, talking face generation has made remarkable progress. However, existing diffusion-based methods still require task-specific fine-tuning and large-scale audiovisual datasets, resulting in high computational costs that hinder the scalability and accessibility of diffusion-based approaches across the research community. To address this, we propose a fine-tuning-free paradigm that directly performs talking face generation using the pretrained weights of Stable Diffusion and IP-Adapter. This backbone leverages the visual embedding capability of IP-Adapter to mine lip-related semantics from the pretrained Stable Diffusion. To address the challenges of identity drift, synchronization errors, and temporal instability, we also design three trainable-parameterfree components: (1) the Structurist, which explicitly disentangles and reassembles lip and appearance features to mitigate identity drift and appearance distortion; (2) the Structure Controller, which adaptively refines embeddings based on quasi-monotonic motion trends for precise lip synchronization; and (3) the Noise Sensor, which introduces Gaussian prior to detect and suppress flicker and jitter artifacts and enhance temporal consistency. Experimental results show that our method outperforms existing SOTA approaches in both lip-sync accuracy (at least 0.16 gain in PCLD) and visual fidelity (at least 0.7 improvement in FID), establishing a novel fine-tuning-free diffusion framework for talking face generation.

---

## Repository Layout

```text
freetalkdiff/
|-- configs/
|   `-- inference.yaml                              # Main runtime configuration
|-- gfpgan/                                         # GFPGAN weight
|   `-- weights/
|       |-- detection_Resnet50_Final.pth
|       `-- parsing_parsenet.pth
|-- modules/             
|   |-- face_enhancer/                              # GFPGAN and RealESRGAN weight
|   |   |-- GFPGANv1.4.pth             
|   |   `-- RealESRGAN_x4plus.pth             
|   |-- mediapipe/                                  # MediaPipe local assets
|   |   |-- canonical_face_model.obj   
|   |   `-- selfie_multiclass_256x256.tflite   
|   |-- face_alignment.py                           # InsightFace alignment and recovery utilities
|   |-- face_enhancer.py                            # GFPGAN/RealESRGAN restoration
|   |-- masker.py                                   # MediaPipe segmentation-based mouth/face masks
|   |-- mediapipe_segmenter.py                      # Selfie segmentation wrapper
|   |-- noise_sensor.py                             # The proposed module
|   |-- structure_controller.py                     # The proposed module
|   |-- structurist.py                              # The proposed module
|-- pipelines/   
|   |-- sd15_inpaint_ipadapter_faceid_pipeline.py   # Diffusers backblone
|   `-- freetalkdiff.py                             # Full FreeTalkDiff pipeline
|-- scripts/             
|   `-- inference.py                                # End-to-end demo script
```

## Environment Setup

Create an environment:

```bash
conda create -n attack_talker python=3.10 -y
conda activate attack_talker
```

Install PyTorch for your CUDA version first. For example, choose the command recommended by the official PyTorch selector for your driver/runtime:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining runtime dependencies from the project requirements file:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file is a minimal runtime dependency list for the main inference pipeline.

Notes:

- Match the PyTorch and CUDA versions to your own GPU environment.
- Match `onnxruntime-gpu` to your CUDA/cuDNN runtime as well.
- `torchvision.io.read_video` and `write_video` require a working video backend such as FFmpeg.
- InsightFace may download its `buffalo_l` model package into the local InsightFace cache on first use.

## Model And Asset Checklist

FreeTalkDiff loads several models automatically from Hugging Face when the pipeline starts:

- `stable-diffusion-v1-5/stable-diffusion-inpainting`
- `h94/IP-Adapter`
- `h94/IP-Adapter-FaceID`
- `OedoSoldier/detail-tweaker-lora`

The repo/config also expects these local assets:

- `gfpgan/weights/detection_Resnet50_Final.pth`
- `gfpgan/weights/parsing_parsenet.pth`
- `modules/mediapipe/selfie_multiclass_256x256.tflite`
- `modules/mediapipe/canonical_face_model.obj`
- `modules/face_enhancer/RealESRGAN_x4plus.pth`
- `modules/face_enhancer/GFPGANv1.4.pth`
- InsightFace `buffalo_l` model cache, downloaded by InsightFace if missing

You can download them from [Google Driven](https://drive.google.com/drive/folders/1nAV-4Fohf06X7Zhj-okh37fcXVD-KUZa?usp=sharing).

## Quick Start

Run inference with an identity video, a driven video, and an output path:

```bash
python scripts/inference.py \
  --cfg_path configs/inference.yaml \
  --id_video path/to/id_video.mp4 \
  --driven_video path/to/driven_video.mp4 \
  --output_video path/to/output.mp4
```

Arguments:

- `--cfg_path`: inference config path. Defaults to `configs/inference.yaml`.
- `--id_video`: identity and appearance source video.
- `--driven_video`: mouth motion driving video.
- `--output_video`: output video save path.

## Configuration

Important fields in `configs/inference.yaml`:

| Field | Default | Meaning |
| --- | --- | --- |
| `freetalkdiff.setup.det_size` | `256` | Face detection/alignment size. |
| `freetalkdiff.setup.scale` | `1.0` | IP-Adapter scale. |
| `freetalkdiff.setup.torch_dtype` | `float16` | Torch dtype used for diffusion models. |
| `freetalkdiff.setup.variant` | `fp16` | Hugging Face model variant. |
| `freetalkdiff.setup.device` | `cuda` | Runtime device. |
| `freetalkdiff.setup.seed` | `42` | Generator seed for repeatability. |
| `freetalkdiff.setup.proxy_port` | `7890` | Local proxy port and Hugging Face mirror switch. |
| `freetalkdiff.inference.guidance_scale` | `7.5` | Classifier-free guidance scale. |
| `freetalkdiff.inference.num_inference_steps` | `50` | Diffusion sampling steps. |
| `freetalkdiff.inference.strength` | `1.0` | Inpainting denoising strength. |
| `freetalkdiff.inference.clip_len` | `5` | Number of frames processed per diffusion clip. |

When `proxy_port` is not `null`, the code sets `HF_ENDPOINT=https://hf-mirror.com` and configures `HTTP_PROXY`/`HTTPS_PROXY` to `http://127.0.0.1:{proxy_port}`. Set `proxy_port: null` if you do not want this behavior.

## Troubleshooting

**Hugging Face downloads fail**

- Check network access to Hugging Face or the configured mirror.
- If you do not use a local proxy, set `freetalkdiff.setup.proxy_port: null`.
- If you do use a proxy, make sure it is listening on the configured port.

**CUDA or ONNX Runtime provider errors**

- Install `onnxruntime-gpu` that matches your CUDA runtime.
- Confirm PyTorch sees your GPU with `python -c "import torch; print(torch.cuda.is_available())"`.
- If you need CPU debugging, change `device` and provider assumptions carefully; the current pipeline is written for CUDA/fp16 execution.

**No face detected**

- Use clear, frontal source and driven videos.
- Ensure both videos contain a detectable face in the first frames.
- The pipeline falls back to previous valid embeddings for some per-frame misses, but the first valid frame must contain a face.

**Missing local assets**

- Verify the MediaPipe files listed in the asset checklist.
- Verify GFPGAN and RealESRGAN weights before enabling `FaceEnhancer`.
- Confirm InsightFace can download or find `buffalo_l`.

**Video read/write errors**

- Install FFmpeg and ensure it is visible to the Python environment.
- Try writing to another codec/container if `torchvision.io.write_video` fails on your platform.

## Discussion

[AnimateDiff](https://github.com/guoyww/AnimateDiff) extends SD 1.5 by introducing pluggable motion modules and motion LoRA, enabling coherent video generation. This design does not compromise the compatibility with IP-Adapter. Consequently, AnimateDiff can be paired with IP-Adapter to form a \textit{seemingly} feasible fine-tuning-free backbone for talking face generation. However, the current IP-Adapter only provides a shared lip reference for the entire input video clip in [Diffusers](https://github.com/huggingface/diffusers), rather than assigning distinct lip features to each frame. This limitation causes the `AnimateDiff + IP-Adapter` backbone to produce static lip shape across frames, preventing it from achieving fine-tuning-free talking face generation. Looking ahead, with the continued evolution of the AnimateDiff, IP-Adapter, and Diffusers communities, as well as the increasing modularity and openness of these frameworks, we believe this combined backbone holds great potential to advance fine-tuning-free talking face generation.

<!-- ## Citation

```bibtex
@article{TODO_FreeTalkDiff,
  title   = {TODO: FreeTalkDiff paper title},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
``` -->

## License

The project license is `Apache-2.0`.

Please also check the licenses and usage terms of the upstream models and assets used by this project.

## Acknowledgements

This project builds on excellent open-source work from the [Diffusers](https://github.com/huggingface/diffusers), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), [InsightFace](https://github.com/deepinsight/insightface), [MediaPipe](https://github.com/google-ai-edge/mediapipe), [GFPGAN](https://github.com/TencentARC/GFPGAN), and [RealESRGAN](https://github.com/xinntao/real-esrgan) communities.
