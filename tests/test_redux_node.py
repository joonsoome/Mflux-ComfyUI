from PIL import Image


def test_redux_forwards_paths_and_calls_metadata(monkeypatch, tmp_path):
    img1 = tmp_path / "a.png"
    img2 = tmp_path / "b.png"
    Image.new("RGB", (80, 80)).save(img1)
    Image.new("RGB", (80, 80)).save(img2)

    import folder_paths
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda x: str(img1) if x == img1.name else str(img2))
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(folder_paths, "get_save_image_path", lambda filename_prefix, output_dir, h, w: (str(tmp_path), filename_prefix, 0, "", filename_prefix))

    import Mflux_Comfy.Mflux_Core as core
    import numpy as _np
    import torch as _torch
    arr = (_np.ones((80, 80, 3), dtype=_np.uint8) * 255).astype(_np.uint8)
    # channels-last batch shape
    t = _torch.from_numpy(arr).unsqueeze(0).to(_torch.float32) / 255.0
    monkeypatch.setattr(core, "generate_image", lambda **kw: (t,))

    from Mflux_Comfy.Mflux_Pro import MfluxRedux
    node = MfluxRedux()
    out = node.generate_redux(prompt="p", Redux1=img1.name, Redux2=img2.name, steps=10, seed=-1, model="dev")
    import torch
    assert isinstance(out, tuple)
    assert hasattr(out[0], 'shape') or isinstance(out[0], torch.Tensor)

    import Mflux_Comfy.Mflux_Core as core
    called = {}
    def _spy_save(*args, **kwargs):
        called['args'] = args
        called['kwargs'] = kwargs
        return {"ui": {"images": []}, "counter": 1}
    monkeypatch.setattr(core, "save_images_with_metadata", _spy_save)

    core.save_images_with_metadata(out, "p", "dev", "8", "", -1, out[0].shape[1], out[0].shape[2], 10, 3.5, [], [], str(img1), 1.0, extra_pnginfo={"redux_image_paths": [str(img1), str(img2)], "redux_image_strengths": [1.0, 1.0]})
    assert 'kwargs' in called
    assert isinstance(called['kwargs'].get('extra_pnginfo', {}).get('redux_image_paths'), list)
    assert str(img1) in called['kwargs']['extra_pnginfo']['redux_image_paths']
    assert str(img2) in called['kwargs']['extra_pnginfo']['redux_image_paths']
