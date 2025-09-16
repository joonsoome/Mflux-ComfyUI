import os
import json
from types import SimpleNamespace
import pytest
from Mflux_Comfy.Mflux_Air import QuickMfluxNode


def test_quicknode_forwards_vae_flags(monkeypatch, tmp_path):
    # Prepare a dummy generate_image stub that asserts it receives vae flags
    captured = {}

    def fake_generate_image(prompt, model, seed, width, height, steps, guidance, quantize, metadata, Local_model, img2img, Loras, ControlNet, base_model=None, low_ram=False, vae_tiling=False, vae_tiling_split="horizontal"):
        captured['generate_kwargs'] = {
            'vae_tiling': vae_tiling,
            'vae_tiling_split': vae_tiling_split,
        }
        # Return a trivial image tensor tuple expected by Quick node
        import numpy as np
        import torch
        arr = np.zeros((1, 16, 16, 3), dtype=np.float32)
        t = torch.from_numpy(arr)
        return (t,)

    def fake_save_images_with_metadata(**kwargs):
        captured['save_kwargs'] = kwargs
        # Simulate return value
        return {"ui": {"images": []}, "counter": 1}

    # QuickMfluxNode imports generate_image and save_images_with_metadata into its module
    monkeypatch.setattr('Mflux_Comfy.Mflux_Air.generate_image', fake_generate_image)
    monkeypatch.setattr('Mflux_Comfy.Mflux_Air.save_images_with_metadata', fake_save_images_with_metadata)

    # Call QuickMfluxNode.generate with vae flags enabled
    node = QuickMfluxNode()
    img_tuple = node.generate(
        prompt="a test",
        model="schnell",
        seed=123,
        width=128,
        height=128,
        steps=10,
        guidance=3.5,
        quantize="8",
        metadata=True,
        Local_model="",
        img2img=None,
        Loras=None,
        ControlNet=None,
        base_model="dev",
        low_ram=False,
        full_prompt=None,
        extra_pnginfo=None,
        size_preset="Custom",
        apply_size_preset=True,
        quality_preset="Custom",
        apply_quality_preset=True,
        randomize_seed=False,
        vae_tiling=True,
        vae_tiling_split="vertical",
    )

    # Assert generate_image received the flags
    assert captured.get('generate_kwargs', {}).get('vae_tiling') is True
    assert captured.get('generate_kwargs', {}).get('vae_tiling_split') == 'vertical'

    # Assert save_images_with_metadata was called with vae flags
    assert 'save_kwargs' in captured
    assert captured['save_kwargs'].get('vae_tiling') is True
    assert captured['save_kwargs'].get('vae_tiling_split') == 'vertical'
