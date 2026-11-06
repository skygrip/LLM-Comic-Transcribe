"""Comic page extraction from EPUB and CBZ/ZIP archives."""

import io
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif")


def extract_comic_pages(archive_path: Path | str) -> list[tuple[int, str, bytes]]:
    """Extract comic page image bytes in exact spine (reading) order from an EPUB,
    or sorted natural order from a CBZ / ZIP archive.

    Returns:
        List of (page_num, image_path_in_zip, image_bytes) sequentially 1-indexed.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"File not found: {archive_path}")

    raw_pages = []
    with zipfile.ZipFile(archive_path, "r") as z:
        # Search for OPF file (EPUB package file)
        opf_path = None
        for name in z.namelist():
            if name.lower().endswith(".opf"):
                opf_path = name
                break

        def add_page(img_path: str, img_data: bytes):
            # Skip consecutive identical byte duplicates (e.g. Calibre duplicate cover clone)
            if raw_pages and raw_pages[-1][1] == img_data:
                return
            raw_pages.append((img_path, img_data))

        if opf_path:
            # EPUB format: parse OPF manifest and spine for strict reading order
            opf_dir = Path(opf_path).parent.as_posix()
            opf_xml = z.read(opf_path).decode("utf-8", errors="ignore")
            root = ET.fromstring(opf_xml)

            # Build manifest map: id -> full zip path
            manifest = {}
            for el in root.iter():
                if el.tag.endswith("item") and "id" in el.attrib and "href" in el.attrib:
                    item_id = el.attrib["id"]
                    href = el.attrib["href"]
                    # Resolve relative path from opf location
                    if opf_dir and opf_dir != ".":
                        full_href = f"{opf_dir}/{href}"
                    else:
                        full_href = href
                    manifest[item_id] = full_href

            # Read spine order
            spine_ids = [
                el.attrib["idref"]
                for el in root.iter()
                if el.tag.endswith("itemref") and "idref" in el.attrib
            ]

            # For each spine entry, find the embedded image(s) or direct image reference
            for sid in spine_ids:
                item_path = manifest.get(sid)
                if not item_path:
                    continue

                # Case A: Spine item is directly an image
                if item_path.lower().endswith(IMG_EXTENSIONS) and item_path in z.namelist():
                    add_page(item_path, z.read(item_path))
                    continue

                # Case B: Spine item is an XHTML/HTML document containing image(s)
                try:
                    xhtml_content = z.read(item_path).decode("utf-8", errors="ignore")
                    img_src_matches = re.findall(
                        r'<img[^>]+src=["\']([^"\']+)["\']', xhtml_content, re.IGNORECASE
                    )
                    if not img_src_matches:
                        img_src_matches = re.findall(
                            r'<image[^>]+(?:href|xlink:href)=["\']([^"\']+)["\']',
                            xhtml_content,
                            re.IGNORECASE,
                        )

                    for rel_img in img_src_matches:
                        rel_img = urllib.parse.unquote(rel_img).replace("\\", "/")
                        xhtml_parent = Path(item_path).parent.as_posix()
                        if xhtml_parent and xhtml_parent != ".":
                            normalized_parts = []
                            for part in f"{xhtml_parent}/{rel_img}".split("/"):
                                if part == "..":
                                    if normalized_parts:
                                        normalized_parts.pop()
                                elif part and part != ".":
                                    normalized_parts.append(part)
                            target_img_path = "/".join(normalized_parts)
                        else:
                            target_img_path = rel_img

                        if target_img_path in z.namelist():
                            add_page(target_img_path, z.read(target_img_path))
                except Exception as e:
                    print(f"Warning: Failed reading spine page {item_path}: {e}", file=sys.stderr)

        # Fallback or CBZ/ZIP handling: if not an EPUB or no spine pages mapped to images
        if not raw_pages:
            raw_images = [
                n
                for n in z.namelist()
                if n.lower().endswith(IMG_EXTENSIONS) and not n.startswith("__MACOSX/")
            ]
            # Natural sort (e.g. page_1, page_2, ..., page_10)
            raw_images.sort(
                key=lambda s: [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]
            )
            for r_img in raw_images:
                add_page(r_img, z.read(r_img))

    if not raw_pages:
        raise ValueError(f"No comic page images found in archive: {archive_path}")

    # Detect and drop synthetic Calibre/KCC duplicate cover prepended to Page 1
    if len(raw_pages) >= 2:
        img1_bytes = raw_pages[0][1]
        img2_bytes = raw_pages[1][1]
        is_dup = False
        if img1_bytes == img2_bytes:
            is_dup = True
        else:
            try:
                # Fast 32x32 perceptual check to catch format conversion (e.g. cover.jpeg vs 0 - 001.webp)
                im1 = Image.open(io.BytesIO(img1_bytes)).convert("L").resize((32, 32))
                im2 = Image.open(io.BytesIO(img2_bytes)).convert("L").resize((32, 32))
                diff = sum(abs(a - b) for a, b in zip(im1.tobytes(), im2.tobytes(), strict=True)) / 1024.0
                if diff < 3.0:
                    is_dup = True
            except Exception:
                pass
        if is_dup:
            raw_pages.pop(0)

    # Assign clean, sequential 1-based page numbers: 1, 2, ..., N
    return [(idx + 1, path, data) for idx, (path, data) in enumerate(raw_pages)]
