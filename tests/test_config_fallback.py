from PIL import Image
import os


def test_generate_config_fallback(monkeypatch, tmp_path):
    # Create a tiny image to act as input
    img = tmp_path / "in.png"
    Image.new("RGB", (16, 16)).save(img)

    import folder_paths
    # annotated filepath returns our tmp file
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda x: str(img))
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(folder_paths, "get_save_image_path", lambda filename_prefix, output_dir, h, w: (str(tmp_path), filename_prefix, 0, "", filename_prefix))

    import Mflux_Comfy.Mflux_Core as core

    # Monkeypatch Config in the module to raise TypeError on first call, then succeed
    class DummyConfig:
        def __init__(self, **kwargs):
            # Simulate rejecting unknown keys on first attempt
            if not hasattr(DummyConfig, "tried"):
                DummyConfig.tried = True
                raise TypeError("unexpected keyword argument 'controlnet_cond'")

    monkeypatch.setattr(core, "Config", DummyConfig)

    # Monkeypatch flux generation to return a simple numpy array wrapped in torch tensor
    class DummyFlux:
        def generate_image(self, seed, prompt, config):
            import numpy as np
            import torch
            arr = (np.ones((16, 16, 3), dtype=np.float32) * 255.0)
            t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
            return t

    monkeypatch.setattr(core, "load_or_create_flux", lambda *a, **k: DummyFlux())

    # Call generate_image with an extra Phase-2 kwarg that will trigger the TypeError path
    result = core.generate_image(prompt="p", model="dev", seed=-1, width=16, height=16, steps=1, guidance=1.0, quantize="8", metadata=True, Local_model="", image=type("X", (), {"image_path": str(img), "image_strength": 1.0})(), masked_image_path="/nonexistent/path")

    # Should return a tuple with a tensor-like first element
    assert isinstance(result, tuple)
    assert len(result) == 1
