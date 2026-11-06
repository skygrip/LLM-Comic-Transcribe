"""Unit tests for VLM utilities and image optimizations."""

import io

from PIL import Image

from llm_comic_transcribe.vlm import get_image_mime, optimize_image_for_vlm


def test_get_image_mime():
    png_header = b"\x89PNG\r\n\x1a\n"
    webp_header = b"RIFF\x00\x00\x00\x00WEBPVP8 "
    jpeg_header = b"\xff\xd8\xff\xe0"

    assert get_image_mime(png_header) == "image/png"
    assert get_image_mime(webp_header) == "image/webp"
    assert get_image_mime(jpeg_header) == "image/jpeg"


def test_optimize_image_for_vlm():
    # Create large 3000x3000 image
    im = Image.new("RGB", (3000, 3000), color=(128, 128, 128))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    large_bytes = buf.getvalue()

    opt_bytes, mime = optimize_image_for_vlm(large_bytes, max_dim=1000)
    assert mime == "image/jpeg"

    # Verify resized dimension <= 1000
    res_im = Image.open(io.BytesIO(opt_bytes))
    assert max(res_im.size) <= 1000
