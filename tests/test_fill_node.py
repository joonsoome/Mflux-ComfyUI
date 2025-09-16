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
    called = {}

    def fake_generate(**kwargs):
        called['generate'] = kwargs
        return ("IMG",)

    def fake_save(images, *args, **kwargs):
        called['save'] = True

    monkeypatch.setattr(core, "generate_image", fake_generate)
    monkeypatch.setattr(core, "save_images_with_metadata", fake_save)

    from Mflux_Comfy.Mflux_Pro import MfluxFill
    node = MfluxFill()
    out = node.generate_fill(prompt="p", image=src.name, masked_image=mask.name, steps=10, seed=-1, model="dev")
    assert out == ("IMG",)
    # Verify metadata JSON was written and contains masked_image_path
    mflux_dir = tmp_path / "MFlux"
    json_files = list(mflux_dir.glob("*.json"))
    assert len(json_files) >= 1
    import json
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    assert data.get("masked_image_path") == str(mask)
