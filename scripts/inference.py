import torch
import argparse
from omegaconf import OmegaConf
from torchvision.io import read_video, write_video

from pipelines import FreeTalkDiff
from modules import FaceAlignment, FaceEnhancer


def main(args):
    cfg = OmegaConf.load(args.cfg_path)
    setup = OmegaConf.to_container(cfg.freetalkdiff.setup, resolve=True)
    setup["torch_dtype"] = getattr(torch, setup["torch_dtype"])
    freetalkdiff = FreeTalkDiff(cfg, **setup)
    fa = FaceAlignment(freetalkdiff.app, freetalkdiff.masker, freetalkdiff.det_size)
    face_enhancer = FaceEnhancer(**cfg.face_enhancer)
    
    id_video_ori, _, _ = read_video(args.id_video, output_format="TCHW")
    driven_video_ori, _, _ = read_video(args.driven_video, output_format="TCHW")
    num_frames = min(id_video_ori.shape[0], driven_video_ori.shape[0])
    id_video_ori = id_video_ori[:num_frames]
    driven_video_ori = driven_video_ori[:num_frames]
    id_video, Ms = fa.align_and_crop(id_video_ori)
    driven_video, _ = fa.align_and_crop(driven_video_ori)
    
    out = freetalkdiff(id_video, driven_video, **cfg.freetalkdiff.inference)
    
    out = face_enhancer(out, out_size=freetalkdiff.det_size)
    out = fa.recover(out, Ms, bg_frames=id_video_ori)
    out = out.permute(0, 2, 3, 1)
    write_video(args.output_video, out, fps=25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", default="configs/inference.yaml", help="The filepath of config yaml")
    parser.add_argument("--id_video", default="resources/xxx.mp4", help="The filepath of id video")
    parser.add_argument("--driven_video", default="resources/xxx.mp4", help="The filepath of driven video")
    parser.add_argument("--output_video", default="resources/FreeTalkDiff.mp4", help="The savepath of output video")
    args = parser.parse_args()
    
    main(args)
