"""
Strip optimizer states & GAN discriminators from a StyleTTS2 Stage 2 checkpoint
to produce a lightweight inference-only model (~4x smaller).
"""

import argparse
import os
import torch
import yaml

INFERENCE_MODULES = [
    "bert", "bert_encoder", "predictor", "decoder",
    "text_encoder", "predictor_encoder", "style_encoder", "diffusion",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export lightweight inference checkpoint.")
    parser.add_argument("--ckpt", required=True, help="Path to Stage 2 epoch_2nd_*.pth checkpoint")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--out", default="kokoro_release.pth", help="Output lightweight .pth path")
    args = parser.parse_args()

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    raw_net = raw["net"]
    clean_net = {}
    for key in INFERENCE_MODULES:
        if key in raw_net:
            clean_net[key] = {
                (k[7:] if k.startswith("module.") else k): v.detach().cpu()
                for k, v in raw_net[key].items()
            }

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    torch.save(
        {
            "net": clean_net,
            "model_params": cfg["model_params"],
            "defaults": {"alpha": 0.0, "beta": 0.55, "embedding_scale": 1.15, "diffusion_steps": 15},
        },
        args.out,
    )
    src_mb = os.path.getsize(args.ckpt) / 1e6
    dst_mb = os.path.getsize(args.out) / 1e6
    print(f"✅ Exported {args.out}: {src_mb:.1f} MB -> {dst_mb:.1f} MB")


if __name__ == "__main__":
    main()
