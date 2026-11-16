"""LoRA fine-tuning subsystem (PLANNING.md Task 5).

First pass only covers ViT-attention backbones (BiomedCLIP, MedSigLIP) via
`peft`'s standard `LoraConfig` -- Merlin's 3-D I3D-ResNet-152 is
convolutional and needs a different adapter mechanism (a small conv
bottleneck, not built here); MedImageInsights isn't wired in yet either.
See `finetune/lora.py` and RESULTS.md for what's actually been run.
"""
