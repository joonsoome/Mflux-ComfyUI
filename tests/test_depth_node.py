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
    captured = {}
    monkeypatch.setattr(core, "generate_image", lambda **kw: ("DIMG",))
    monkeypatch.setattr(core, "save_images_with_metadata", lambda *a, **k: captured.update({"saved": True}))

    from Mflux_Comfy.Mflux_Pro import MfluxDepth
    node = MfluxDepth()
    out = node.generate_depth(prompt="p", image=src.name, steps=10, seed=-1, model="dev", depth_image=depth.name)
    assert out == ("DIMG",)
    # Verify metadata JSON exists and contains depth_image_path
    mflux_dir = tmp_path / "MFlux"
    json_files = list(mflux_dir.glob("*.json"))
    assert len(json_files) >= 1
    import json
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    assert data.get("depth_image_path") == str(depth)
