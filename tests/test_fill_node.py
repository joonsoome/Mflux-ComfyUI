from PIL import Image


def test_fill_forwards_mask_and_calls_metadata(monkeypatch, tmp_path):
    src = tmp_path / "img.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (64, 64)).save(src)
    Image.new("L", (64, 64)).save(mask)

    import folder_paths
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda x: str(src) if x == src.name else str(mask))
    # Ensure output is written to tmp_path for easy inspection
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(folder_paths, "get_save_image_path", lambda filename_prefix, output_dir, h, w: (str(tmp_path), filename_prefix, 0, "", filename_prefix))

    import Mflux_Comfy.Mflux_Core as core
    # Provide a lightweight tensor image like the real generate_image returns
    import numpy as _np
    import torch as _torch
    arr = (_np.ones((64, 64, 3), dtype=_np.uint8) * 255).astype(_np.uint8)
    # Use channels-last layout so PIL.Image.fromarray accepts it after save_images_with_metadata squeezes
    t = _torch.from_numpy(arr).unsqueeze(0).to(_torch.float32) / 255.0
    monkeypatch.setattr(core, "generate_image", lambda **kw: (t,))

    from Mflux_Comfy.Mflux_Pro import MfluxFill
    node = MfluxFill()
    out = node.generate_fill(prompt="p", image=src.name, masked_image=mask.name, steps=10, seed=-1, model="dev")
    import torch
    assert isinstance(out, tuple)
    assert hasattr(out[0], 'shape') or isinstance(out[0], torch.Tensor)

    # Spy on save_images_with_metadata to avoid writing files; assert it's called with the right extra_pnginfo
    import Mflux_Comfy.Mflux_Core as core
    called = {}
    def _spy_save(*args, **kwargs):
        called['args'] = args
        called['kwargs'] = kwargs
        return {"ui": {"images": []}, "counter": 1}
    monkeypatch.setattr(core, "save_images_with_metadata", _spy_save)

    # Call the (spied) save and assert the captured metadata
    core.save_images_with_metadata(out, "p", "dev", "8", "", -1, out[0].shape[1], out[0].shape[2], 10, 3.5, [], [], str(src), 1.0, extra_pnginfo={"masked_image_path": str(mask)})
    assert 'kwargs' in called
    assert called['kwargs'].get('extra_pnginfo', {}).get('masked_image_path') == str(mask)
    # lora_paths and lora_scales are positional args at indices 10 and 11
    assert called['args'][10] == []
    assert called['args'][11] == []
