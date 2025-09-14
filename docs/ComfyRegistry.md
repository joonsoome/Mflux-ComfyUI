# Comfy Registry — Submission Notes

Short description:
> ComfyUI nodes for mflux 0.10.x (Apple Silicon/MLX). Quick txt2img/img2img with LoRA and ControlNet canny, HF repo support, and metadata saving; keeps legacy graph compatibility.

PublisherId: joonsoome
DisplayName: Mflux-ComfyUI
Icon: assets/icon.svg

Screenshots (recommended):
- examples/Air.png (text2img)
- examples/Air_img2img.png (img2img)
- examples/Pro_Loras.png (LoRA)
- examples/Pro_ControlNet.png (ControlNet)
- examples/Mflux_Metadata.png (metadata)

Notes:
- Backend requires mflux >= 0.10.0
- Recommend MLX >= 0.27.0 on Apple Silicon
- Quantize choices: None, 3, 4, 5, 6, 8 (default 8)
- LoRAs require quantize=8
