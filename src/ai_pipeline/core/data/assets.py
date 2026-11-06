"""Immutable, content-addressed asset store.

Layout under ``artifacts/<project>/assets/``::

    ab/abcdef...<sha256>.jpg      the bytes, written once, chmod 0444
    ab/abcdef...<sha256>.json     sidecar metadata (dimensions, original names, import ids)

The asset id *is* the SHA-256 of the file content, so importing the same image twice (from
any path, under any filename) yields the same asset and no second copy. Only the sidecar's
provenance lists (``original_filenames``, ``import_ids``) grow on re-import; the bytes never
change.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import stat
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from ai_pipeline.core.errors import NotFoundError, OperationError
from ai_pipeline.core.ids import utc_now_iso
from ai_pipeline.core.store import read_json, write_json_atomic

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH


class AssetInfo(BaseModel):
    asset_id: str
    ext: str
    width: int
    height: int
    size_bytes: int
    format: str
    original_filenames: list[str] = Field(default_factory=list)
    import_ids: list[str] = Field(default_factory=list)
    first_imported_at: str

    @property
    def filename(self) -> str:
        return f"{self.asset_id}{self.ext}"

    @property
    def stems(self) -> set[str]:
        return {Path(original_basename(n)).stem for n in self.original_filenames}


def original_basename(name: str) -> str:
    """File name without directories. Archives made on Windows can carry ``\\`` separators
    that were recorded verbatim as part of the original filename."""
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_image(path: Path) -> tuple[int, int, str]:
    """Return ``(width, height, format)``; raise ``ValueError`` for unreadable/corrupt files."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:  # verify() invalidates the handle; reopen for size
            width, height = im.size
            fmt = (im.format or "").upper()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"not a readable image: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError("image has zero dimension")
    return width, height, fmt


def _make_read_only(path: Path) -> None:
    # Filesystems without permission bits: immutability is best-effort there.
    with contextlib.suppress(OSError):
        os.chmod(path, READ_ONLY)


class AssetStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- paths ----------------------------------------------------------------------

    def _shard(self, asset_id: str) -> Path:
        return self.root / asset_id[:2]

    def sidecar_path(self, asset_id: str) -> Path:
        return self._shard(asset_id) / f"{asset_id}.json"

    def file_path(self, info: AssetInfo) -> Path:
        return self._shard(info.asset_id) / info.filename

    # -- queries --------------------------------------------------------------------

    def has(self, asset_id: str) -> bool:
        return self.sidecar_path(asset_id).is_file()

    def info(self, asset_id: str) -> AssetInfo:
        p = self.sidecar_path(asset_id)
        if not p.is_file():
            raise NotFoundError(
                f"asset {asset_id!r} not found",
                code="ASSET_NOT_FOUND",
                details={"asset_id": asset_id, "path": str(p)},
            )
        return AssetInfo.model_validate(read_json(p))

    def all_infos(self) -> list[AssetInfo]:
        if not self.root.is_dir():
            return []
        return sorted(
            (AssetInfo.model_validate(read_json(p)) for p in self.root.glob("??/*.json")),
            key=lambda a: a.asset_id,
        )

    def count(self) -> int:
        return len(self.all_infos())

    # -- mutation (append-only) ------------------------------------------------------

    def add(self, src: Path, *, original_filename: str, import_id: str) -> tuple[AssetInfo, bool]:
        """Add ``src`` to the store. Returns ``(info, created)``.

        ``created`` is False when identical bytes were already present; in that case only the
        sidecar provenance is extended. Raises ``ValueError`` for non-image/corrupt files.
        """
        width, height, fmt = probe_image(src)
        asset_id = sha256_file(src)
        ext = src.suffix.lower()
        if ext == ".jpeg":
            ext = ".jpg"

        if self.has(asset_id):
            info = self.info(asset_id)
            changed = False
            if original_filename not in info.original_filenames:
                info.original_filenames.append(original_filename)
                changed = True
            if import_id not in info.import_ids:
                info.import_ids.append(import_id)
                changed = True
            if changed:
                write_json_atomic(self.sidecar_path(asset_id), info.model_dump())
            return info, False

        info = AssetInfo(
            asset_id=asset_id,
            ext=ext,
            width=width,
            height=height,
            size_bytes=src.stat().st_size,
            format=fmt,
            original_filenames=[original_filename],
            import_ids=[import_id],
            first_imported_at=utc_now_iso(),
        )
        dest = self.file_path(info)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        shutil.copyfile(src, tmp)
        if sha256_file(tmp) != asset_id:
            tmp.unlink(missing_ok=True)
            raise OperationError(
                f"checksum mismatch while copying {src}",
                code="ASSET_COPY_CORRUPTED",
                details={"source": str(src), "asset_id": asset_id},
                hint="The source file changed during import or the disk is unreliable. Retry.",
            )
        os.replace(tmp, dest)
        _make_read_only(dest)
        write_json_atomic(self.sidecar_path(asset_id), info.model_dump())
        return info, True

    def verify(self, info: AssetInfo) -> bool:
        """Re-hash the stored bytes; True when they still match the asset id."""
        p = self.file_path(info)
        return p.is_file() and sha256_file(p) == info.asset_id
