import os
from PIL import Image

# Import the helper directly; underscore prefix is okay for tests
from Mflux_Comfy.Mflux_Pro import _make_oriented_copy


def test_exif_orientation_transpose(tmp_path):
    # Create a wide test image
    src = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 50), color=(128, 64, 32))

    # Write EXIF orientation: 6 means Rotate 90 CW when displaying
    exif = img.getexif()
    exif[274] = 6  # 274 is Orientation tag
    img.save(src, format="JPEG", exif=exif.tobytes())

    new_path, w, h = _make_oriented_copy(str(src))

    assert os.path.exists(new_path)
    # After applying exif_transpose, width/height should be swapped
    assert (w, h) == (50, 100)
