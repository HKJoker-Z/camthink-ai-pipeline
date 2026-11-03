from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from tests.conftest import make_image, make_image_set

from ai_pipeline.core.data.assets import AssetStore, probe_image, sha256_file
from ai_pipeline.core.data.importer import (
    asset_store,
    import_images,
    resolve_sources,
    scoped_asset_infos,
)
from ai_pipeline.core.errors import ConfigError, NotFoundError, OperationError


def test_probe_image_and_corrupt(tmp_path: Path):
    p = make_image(tmp_path / "a.png", 1, size=(10, 7))
    assert probe_image(p) == (10, 7, "PNG")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"\xff\xd8not really a jpeg")
    with pytest.raises(ValueError):
        probe_image(bad)


def test_asset_store_dedups_by_content_and_is_read_only(tmp_path: Path):
    store = AssetStore(tmp_path / "assets")
    img = make_image(tmp_path / "x" / "cat.jpg", 5)
    same_bytes = tmp_path / "y" / "renamed.jpg"
    same_bytes.parent.mkdir()
    same_bytes.write_bytes(img.read_bytes())

    info, created = store.add(img, original_filename="x/cat.jpg", import_id="imp-1")
    assert created and info.asset_id == sha256_file(img)
    assert store.file_path(info).suffix == ".jpg"
    mode = stat.S_IMODE(os.stat(store.file_path(info)).st_mode)
    assert not mode & stat.S_IWUSR, "stored asset must be read-only"

    info2, created2 = store.add(same_bytes, original_filename="y/renamed.jpg", import_id="imp-2")
    assert not created2 and info2.asset_id == info.asset_id
    assert info2.original_filenames == ["x/cat.jpg", "y/renamed.jpg"]
    assert info2.import_ids == ["imp-1", "imp-2"]
    assert store.count() == 1
    assert len(list((tmp_path / "assets").rglob("*.jpg"))) == 1, "no duplicate physical copy"
    assert store.verify(info2)


def test_asset_store_not_found(tmp_path: Path):
    with pytest.raises(NotFoundError):
        AssetStore(tmp_path).info("deadbeef")


def test_import_images_records_everything(demo_project, tmp_path: Path):
    src = tmp_path / "src"
    make_image_set(src, 5)
    make_image(src / "nested" / "deep.png", 99)
    (src / "readme.txt").write_text("not an image")
    (src / "broken.jpg").write_bytes(b"nope")

    rec = import_images(demo_project, [src])
    c = rec["counts"]
    assert c == {
        "files_seen": 8,
        "images_new": 6,
        "images_existing": 0,
        "rejected": 1,
        "skipped_non_image": 1,
    }
    assert rec["rejected"][0]["path"].endswith("broken.jpg")
    assert {a["filename"] for a in rec["assets"]} >= {"img_000.jpg", "nested/deep.png"}
    assert demo_project.store.read("imports", rec["import_id"]) == rec
    assert asset_store(demo_project).count() == 6

    # Re-import: nothing new, still an import event, no new copies.
    rec2 = import_images(demo_project, [src])
    assert rec2["counts"]["images_new"] == 0 and rec2["counts"]["images_existing"] == 6
    assert asset_store(demo_project).count() == 6
    assert demo_project.store.count("imports") == 2

    # Scope by import id.
    assert len(scoped_asset_infos(demo_project, [rec["import_id"]])) == 6
    assert len(scoped_asset_infos(demo_project, None)) == 6
    with pytest.raises(NotFoundError):
        scoped_asset_infos(demo_project, ["imp-nope"])


def test_import_non_recursive_and_single_file(demo_project, tmp_path: Path):
    src = tmp_path / "src"
    make_image_set(src, 2)
    make_image(src / "nested" / "deep.jpg", 7)
    rec = import_images(demo_project, [src], recursive=False)
    assert rec["counts"]["images_new"] == 2
    rec = import_images(demo_project, [src / "nested" / "deep.jpg"])
    assert rec["counts"]["images_new"] == 1 and rec["assets"][0]["filename"] == "deep.jpg"


def test_import_errors(demo_project, tmp_path: Path):
    with pytest.raises(OperationError) as ei:
        import_images(demo_project, [tmp_path / "missing"])
    assert ei.value.code == "IMPORT_SOURCE_MISSING"
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "x.txt").write_text("")
    with pytest.raises(OperationError) as ei:
        import_images(demo_project, [empty])
    assert ei.value.code == "NO_IMAGES_FOUND"


def test_resolve_sources_defaults_to_project_config(demo_project, tmp_path: Path):
    assert resolve_sources(demo_project, None) == [(demo_project.dir / "data/raw").resolve()]
    assert resolve_sources(demo_project, tmp_path / "x") == [(tmp_path / "x").resolve()]
    text = demo_project.config_path.read_text().replace(
        "- type: local_images\n      path: ./data/raw", "- type: rtsp_capture\n      url: rtsp://x"
    )
    demo_project.config_path.write_text(text)
    from ai_pipeline.core.project import Project

    p = Project.load("demo", demo_project.settings)
    with pytest.raises(ConfigError) as ei:
        resolve_sources(p, None)
    assert ei.value.code == "NO_LOCAL_IMAGE_SOURCE"
