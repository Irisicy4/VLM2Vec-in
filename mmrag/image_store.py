"""Read InfoSeek/OVEN images directly out of the M2KR tar archives — no extraction, no inode cost.

The scratch filesystem has a ~1.02M-inode quota that is already ~97% used, so unpacking 780K small
image files is not an option. Instead we index each tar once ({member name -> (offset, size)}, one
JSON per tar) and pread() image bytes on demand.

    store = TarImageStore(["/…/infoseek_val_images.tar", "/…/infoseek_train_images.tar"])
    img = store.get("oven_05001186")   # PIL.Image (RGB)
"""

import io
import json
import os
import tarfile

from PIL import Image


def build_tar_index(tar_path: str, index_path: str = None) -> dict:
    """Index a tar's file members: basename-without-extension -> (offset_data, size). Saved as JSON."""
    index_path = index_path or tar_path + ".index.json"
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)
    index = {}
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            key = os.path.splitext(os.path.basename(m.name))[0]
            index[key] = (m.offset_data, m.size)
    tmp = index_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(index, f)
    os.replace(tmp, index_path)
    return index


class TarImageStore:
    def __init__(self, tar_paths):
        if isinstance(tar_paths, str):
            tar_paths = [tar_paths]
        self._entries = {}  # key -> (fd_index, offset, size)
        self._fds = []
        for ti, tp in enumerate(tar_paths):
            idx = build_tar_index(tp)
            self._fds.append(os.open(tp, os.O_RDONLY))
            for k, (off, size) in idx.items():
                self._entries.setdefault(k, (ti, off, size))

    def __contains__(self, image_id):
        return image_id in self._entries

    def __len__(self):
        return len(self._entries)

    def get_bytes(self, image_id: str) -> bytes:
        ti, off, size = self._entries[image_id]
        return os.pread(self._fds[ti], size, off)

    def get(self, image_id: str, min_side: int = 28, max_side: int = 1344) -> Image.Image:
        img = Image.open(io.BytesIO(self.get_bytes(image_id)))
        img = img.convert("RGB")
        # Qwen2-VL processors need >=28px on each side; huge images waste vision tokens.
        w, h = img.size
        if min(w, h) < min_side:
            s = min_side / min(w, h)
            img = img.resize((max(min_side, int(w * s)), max(min_side, int(h * s))))
        elif max(w, h) > max_side:
            s = max_side / max(w, h)
            img = img.resize((max(min_side, int(w * s)), max(min_side, int(h * s))))
        return img
