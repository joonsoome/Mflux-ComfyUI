from PIL import Image


def test_depth_forwards_depth_and_calls_metadata(monkeypatch, tmp_path):
    src = tmp_path / "img2.png"
    depth = tmp_path / "depth.png"
    Image.new("RGB", (32, 48)).save(src)
    Image.new("L", (32, 48)).save(depth)

    import folder_paths
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda x: str(src) if x == src.name else str(depth))
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(folder_paths, "get_save_image_path", lambda filename_prefix, output_dir, h, w: (str(tmp_path), filename_prefix, 0, "", filename_prefix))

    import Mflux_Comfy.Mflux_Core as core
    import numpy as _np
    import torch as _torch
    arr = (_np.ones((48, 32, 3), dtype=_np.uint8) * 255).astype(_np.uint8)
    # channels-last batch shape
    t = _torch.from_numpy(arr).unsqueeze(0).to(_torch.float32) / 255.0
    monkeypatch.setattr(core, "generate_image", lambda **kw: (t,))

    from Mflux_Comfy.Mflux_Pro import MfluxDepth
    node = MfluxDepth()
    out = node.generate_depth(prompt="p", image=src.name, steps=10, seed=-1, model="dev", depth_image=depth.name)
    import torch
    assert isinstance(out, tuple)
    assert hasattr(out[0], 'shape') or isinstance(out[0], torch.Tensor)

    # Spy on save_images_with_metadata to avoid writing files
    import Mflux_Comfy.Mflux_Core as core
    called = {}
    def _spy_save(*args, **kwargs):
        called['args'] = args
        called['kwargs'] = kwargs
        return {"ui": {"images": []}, "counter": 1}
    monkeypatch.setattr(core, "save_images_with_metadata", _spy_save)

    core.save_images_with_metadata(out, "p", "dev", "8", "", -1, out[0].shape[1], out[0].shape[2], 10, 3.5, [], [], str(src), 1.0, extra_pnginfo={"depth_image_path": str(depth)})
    assert 'kwargs' in called
    assert called['kwargs'].get('extra_pnginfo', {}).get('depth_image_path') == str(depth)
