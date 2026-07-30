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


class ChainStore:
    """First store that contains the key wins."""

    def __init__(self, stores):
        self.stores = stores

    def __contains__(self, key):
        return any(key in s for s in self.stores)

    def __len__(self):
        return sum(len(s) for s in self.stores)

    def get(self, key, **kw):
        for s in self.stores:
            if key in s:
                return s.get(key, **kw)
        raise KeyError(key)


def open_stores(data_dir, dataset="infoseek"):
    """Image stores for a dataset's query files (only archives that exist are opened)."""
    import os

    if dataset == "infoseek":
        paths = [os.path.join(data_dir, "images/Infoseek", f)
                 for f in ("infoseek_val_images.tar", "infoseek_train_images.tar")]
        return ChainStore([TarImageStore(p) for p in paths if os.path.exists(p)])
    if dataset == "evqa":
        stores = []
        z = os.path.join(data_dir, "images/EVQA/inat.zip")
        if os.path.exists(z):
            stores.append(ZipImageStore(z))          # keys = member paths (id2name values)
        t = os.path.join(data_dir, "images/EVQA/google-landmark.tar")
        if os.path.exists(t):
            stores.append(TarImageStore(t))          # keys = basename w/o ext = landmark id
        return ChainStore(stores)
    raise ValueError(dataset)


class ZipImageStore:
    """Same idea for zip archives (zip has a central directory -> native random access).
    Keys are member paths (e.g. iNat 'train/<category>/<uuid>.jpg'); pass key_fn to remap."""

    def __init__(self, zip_paths, key_fn=None):
        import zipfile

        if isinstance(zip_paths, str):
            zip_paths = [zip_paths]
        self._zfs = [zipfile.ZipFile(p) for p in zip_paths]
        self._entries = {}
        for zi, zf in enumerate(self._zfs):
            for n in zf.namelist():
                if n.endswith("/"):
                    continue
                k = key_fn(n) if key_fn else n
                self._entries.setdefault(k, (zi, n))

    def __contains__(self, key):
        return key in self._entries

    def __len__(self):
        return len(self._entries)

    def get_bytes(self, key):
        zi, name = self._entries[key]
        return self._zfs[zi].read(name)

    def get(self, key, min_side=28, max_side=1344):
        img = Image.open(io.BytesIO(self.get_bytes(key))).convert("RGB")
        w, h = img.size
        if min(w, h) < min_side:
            s = min_side / min(w, h)
            img = img.resize((max(min_side, int(w * s)), max(min_side, int(h * s))))
        elif max(w, h) > max_side:
            s = max_side / max(w, h)
            img = img.resize((max(min_side, int(w * s)), max(min_side, int(h * s))))
        return img


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
