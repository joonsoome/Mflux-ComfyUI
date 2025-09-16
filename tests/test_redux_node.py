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
    monkeypatch.setattr(core, "generate_image", lambda **kw: ("RIMG",))
    captured = {}
    monkeypatch.setattr(core, "save_images_with_metadata", lambda *a, **k: captured.update({"saved": True}))

    from Mflux_Comfy.Mflux_Pro import MfluxRedux
    node = MfluxRedux()
    out = node.generate_redux(prompt="p", Redux1=img1.name, Redux2=img2.name, steps=10, seed=-1, model="dev")
    assert out == ("RIMG",)
    mflux_dir = tmp_path / "MFlux"
    json_files = list(mflux_dir.glob("*.json"))
    assert len(json_files) >= 1
    import json
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    # redux paths should be stored as list of strings
    assert isinstance(data.get("redux_image_paths"), list)
    assert str(img1) in data.get("redux_image_paths")
    assert str(img2) in data.get("redux_image_paths")
