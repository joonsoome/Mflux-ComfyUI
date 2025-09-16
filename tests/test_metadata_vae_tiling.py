import os
import json
import shutil
import tempfile
import numpy as np
import torch
from Mflux_Comfy.Mflux_Core import save_images_with_metadata


def make_dummy_image_tensor():
    # Create a small tensor (1,H,W,C) in float [0,1]
    arr = np.zeros((1, 16, 16, 3), dtype=np.float32)
    arr[0, 0:4, 0:4, :] = 1.0
    # Keep HWC ordering so save_images_with_metadata can call .squeeze() -> (H,W,C)
    t = torch.from_numpy(arr)
    return (t,)


def test_metadata_includes_vae_tiling(tmp_path, monkeypatch):
    # Use a temporary output directory by monkeypatching folder_paths.get_output_directory
    import folder_paths

    tmp_out = tmp_path / "out"
    tmp_out.mkdir()

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_out))

    images = make_dummy_image_tensor()
    prompt = "test"
    model = "dev"
    quantize = "8"
    Local_model = ""
    seed = 42
    height = 16
    width = 16
    steps = 10
    guidance = 3.5
    lora_paths = []
    lora_scales = []
    image_path = None
    image_strength = None

    # Call the save helper with vae_tiling flags set
    res = save_images_with_metadata(
        images=images,
        prompt=prompt,
        model=model,
        quantize=quantize,
        Local_model=Local_model,
        seed=seed,
        height=height,
        width=width,
        steps=steps,
        guidance=guidance,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        image_path=image_path,
        image_strength=image_strength,
        filename_prefix="TEST_MFLUX",
        full_prompt=None,
        extra_pnginfo=None,
        base_model="dev",
        low_ram=False,
        control_image_path=None,
        control_strength=None,
        control_model=None,
        quantize_effective="8-bit",
        vae_tiling=True,
        vae_tiling_split="vertical",
    )

    # Inspect the JSON file written
    outdir = tmp_out / "MFlux"
    files = list(outdir.glob("TEST_MFLUX_*.json"))
    assert files, "No metadata JSON produced"
    # There should be at least one JSON, read the first one
    with open(files[0], 'r') as f:
        js = json.load(f)
    assert js.get("vae_tiling") is True
    assert js.get("vae_tiling_split") == "vertical"

    # Cleanup
    shutil.rmtree(str(tmp_out))
