"""Unit tests for EPUB and CBZ archive extraction."""

import io
import zipfile
from pathlib import Path

from PIL import Image

from llm_comic_transcribe.extractor import extract_comic_pages


def create_dummy_image(color=(255, 0, 0)) -> bytes:
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_extract_cbz(tmp_path: Path):
    cbz_file = tmp_path / "test_comic.cbz"
    img1 = create_dummy_image((255, 0, 0))
    img2 = create_dummy_image((0, 255, 0))
    img3 = create_dummy_image((0, 0, 255))

    with zipfile.ZipFile(cbz_file, "w") as z:
        z.writestr("page_10.jpg", img3)
        z.writestr("page_1.jpg", img1)
        z.writestr("page_2.jpg", img2)

    pages = extract_comic_pages(cbz_file)
    assert len(pages) == 3
    # Check natural sorting: page_1 -> page_2 -> page_10
    assert pages[0][0] == 1
    assert pages[0][1] == "page_1.jpg"
    assert pages[1][0] == 2
    assert pages[1][1] == "page_2.jpg"
    assert pages[2][0] == 3
    assert pages[2][1] == "page_10.jpg"


def test_extract_epub(tmp_path: Path):
    epub_file = tmp_path / "test_comic.epub"
    img1 = create_dummy_image((100, 100, 100))
    img2 = create_dummy_image((200, 200, 200))

    opf_content = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item id="page1" href="images/p1.jpg" media-type="image/jpeg" />
    <item id="page2" href="images/p2.jpg" media-type="image/jpeg" />
  </manifest>
  <spine>
    <itemref idref="page1" />
    <itemref idref="page2" />
  </spine>
</package>
"""
    with zipfile.ZipFile(epub_file, "w") as z:
        z.writestr("OEBPS/content.opf", opf_content)
        z.writestr("OEBPS/images/p1.jpg", img1)
        z.writestr("OEBPS/images/p2.jpg", img2)

    pages = extract_comic_pages(epub_file)
    assert len(pages) == 2
    assert pages[0][0] == 1
    assert pages[0][1] == "OEBPS/images/p1.jpg"
    assert pages[1][0] == 2
    assert pages[1][1] == "OEBPS/images/p2.jpg"
