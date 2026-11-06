"""`label export` review packages and the restricted-SSH `download` helper."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image
from tests.conftest import make_image
from tests.unit.test_cli_m3 import invoke_json

from ai_pipeline.core.data.assets import AssetInfo, original_basename
from ai_pipeline.core.data.importer import import_images, scoped_asset_infos
from ai_pipeline.core.errors import ConfigError, NotFoundError, OperationError
from ai_pipeline.core.labeling.annotations import Annotation, AnnotationSource, BoundingBox
from ai_pipeline.core.labeling.export import clamp_bbox, export_label_set, sequence_hint
from ai_pipeline.core.labeling.labelsets import create_label_set, import_labels
from ai_pipeline.core.project import Project
from ai_pipeline.core.settings import Settings

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "grok_bot"
WINDOWS_BAD = set('<>:"|?*\\')


def _project(workspace: Path, classes: str = "cat,dog") -> Project:
    r, p = invoke_json("project", "create", "demo", "--classes", classes)
    assert r.exit_code == 0, p
    return Project.load("demo", Settings.resolve(cwd=workspace))


def _frames(root: Path, n: int, prefix: str = "cam0_20261106_143807_t") -> list[Path]:
    return [make_image(root / f"{prefix}{i:04d}s.jpg", seed=i + 1) for i in range(1, n + 1)]


def _model_label_set(
    project: Project,
    boxes: dict[str, list[tuple]],
    import_id: str,
    lid: str = "lbl-20260101-000000-abcd",
) -> str:
    """boxes: original file stem -> [(class, x_min, y_min, x_max, y_max, confidence)]."""
    assets = scoped_asset_infos(project, [import_id])
    by_stem = {next(iter(a.stems)): a for a in assets}
    src = AnnotationSource(kind="model", label_set_id=lid, run_id="run-x", provider="fake")
    anns = [
        Annotation(
            asset_id=by_stem[stem].asset_id,
            class_name=c,
            bbox=BoundingBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1),
            confidence=conf,
            source=src,
        )
        for stem, items in boxes.items()
        for (c, x0, y0, x1, y1, conf) in items
    ]
    create_label_set(
        project,
        label_set_id=lid,
        annotations=anns,
        source={"kind": "model", "provider": "fake", "run_id": "run-x", "confidence": 0.25},
        scope_assets=assets,
        import_ids=[import_id],
        labeled_asset_ids={a.asset_id for a in anns},
        run_id="run-x",
    )
    return lid


@pytest.fixture
def exported(workspace: Path, tmp_path: Path):
    project = _project(workspace)
    _frames(tmp_path / "frames" / "images", 4)
    imp = import_images(project, [tmp_path / "frames"])["import_id"]
    lid = _model_label_set(
        project,
        {
            "cam0_20261106_143807_t0001s": [
                ("dog", 0.1, 0.2, 0.3, 0.6, 0.91),
                ("cat", 0.5, 0.5, 0.7, 0.9, None),
            ],
            "cam0_20261106_143807_t0003s": [("cat", -0.05, 0.8, 0.2, 1.1, 0.4)],
        },
        imp,
    )
    rec = export_label_set(project, lid)
    z = zipfile.ZipFile(project.store.root / rec["archive"])
    return project, lid, rec, z, imp


def _read(z: zipfile.ZipFile, name: str) -> str:
    return z.read(f"demo_review/{name}").decode()


# -- conversion, mapping, empties ----------------------------------------------------------------


def test_yolo_coordinates_and_class_mapping(exported):
    _, _, _, z, _ = exported
    assert _read(z, "classes.txt") == "cat\ndog\n"
    lines = [ln.split() for ln in _read(z, "labels/cam0_20261106_143807_t0001s.txt").splitlines()]
    by_cls = {ln[0]: [float(v) for v in ln[1:]] for ln in lines}
    assert set(by_cls) == {"0", "1"} and all(len(ln) == 5 for ln in lines)
    assert by_cls["1"] == pytest.approx([0.2, 0.4, 0.2, 0.4])  # dog = class 1
    assert by_cls["0"] == pytest.approx([0.6, 0.7, 0.2, 0.4])  # cat = class 0
    # an out-of-range proposal is clamped to the image, not dropped
    (clamped,) = [
        ln.split() for ln in _read(z, "labels/cam0_20261106_143807_t0003s.txt").splitlines()
    ]
    assert [float(v) for v in clamped[1:]] == pytest.approx([0.1, 0.9, 0.2, 0.2])


def test_clamp_bbox_reports_changes():
    inside = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2)
    assert clamp_bbox(inside) == (inside, False)
    box, changed = clamp_bbox(BoundingBox(x_min=-1, y_min=0.5, x_max=2, y_max=0.6))
    assert changed and (box.x_min, box.x_max) == (0.0, 1.0)


def test_every_frame_has_a_label_file_and_empties_are_empty(exported):
    _, _, rec, z, _ = exported
    labels = sorted(n for n in z.namelist() if "/labels/" in n)
    assert [Path(n).name for n in labels] == [
        f"cam0_20261106_143807_t{i:04d}s.txt" for i in (1, 2, 3, 4)
    ]
    assert _read(z, "labels/cam0_20261106_143807_t0002s.txt") == ""
    assert not any("/images/" in n for n in z.namelist())  # labels-only by default
    m = json.loads(_read(z, "manifest.json"))
    assert m["counts"] == rec["counts"]
    assert rec["counts"] == {
        "frames": 4,
        "frames_without_boxes": 2,
        "boxes": 3,
        "boxes_clamped": 1,
        "boxes_skipped": 0,
        "boxes_with_confidence": 2,
        "frames_with_boxes": 2,
        "frames_with_timestamp": 0,
    }
    empty = next(f for f in m["frames"] if f["image"] == "cam0_20261106_143807_t0002s.jpg")
    assert empty["boxes"] == [] and empty["label_file"] == "labels/cam0_20261106_143807_t0002s.txt"


def test_manifest_keeps_confidences_and_never_invents_metadata(exported):
    project, lid, _, z, imp = exported
    m = json.loads(_read(z, "manifest.json"))
    assert m["review"]["status"] == "unreviewed" and m["review"]["ground_truth"] is False
    assert m["label_set"]["label_set_id"] == lid and m["label_set"]["scope_import_ids"] == [imp]
    assert m["classes"] == [{"id": 0, "name": "cat"}, {"id": 1, "name": "dog"}]
    f1 = next(f for f in m["frames"] if f["image"].endswith("t0001s.jpg"))
    lines = _read(z, "labels/" + Path(f1["label_file"]).name).splitlines()
    for b in f1["boxes"]:  # manifest box N describes line N of the label file
        assert lines[b["line"] - 1].split()[0] == str(b["class_id"])
    assert sorted((b["class_name"], b["confidence"]) for b in f1["boxes"]) == [
        ("cat", None),
        ("dog", 0.91),
    ]
    assert f1["timestamp"] is None and f1["timestamp_source"] is None
    assert f1["sequence"] == {"group": "cam0_20261106_143807", "offset_s": 1, "source": "filename"}
    assert m["sequences"] == [
        {"group": "cam0_20261106_143807", "frames": 4, "frames_with_boxes": 2}
    ]
    assert f1["asset_id"] in {a.asset_id for a in scoped_asset_infos(project, [imp])}
    # provenance is copied, not guessed: no matching model record -> null version
    assert m["auto_label"]["model_version"] is None
    assert m["auto_label"]["confidence_threshold"] == 0.25


def test_manifest_line_numbers_follow_clamped_order(workspace: Path, tmp_path: Path):
    """Clamping can reorder boxes; manifest line N must still be label-file line N."""
    project = _project(workspace)
    _frames(tmp_path / "f", 1)
    imp = import_images(project, [tmp_path / "f"])["import_id"]
    stem = "cam0_20261106_143807_t0001s"
    lid = _model_label_set(
        project,
        {stem: [("cat", -0.5, 0.6, 0.2, 0.8, 0.9), ("cat", -0.1, 0.1, 0.15, 0.3, 0.3)]},
        imp,
    )
    rec = export_label_set(project, lid)
    z = zipfile.ZipFile(project.store.root / rec["archive"])
    lines = [ln.split() for ln in _read(z, f"labels/{stem}.txt").splitlines()]
    (frame,) = json.loads(_read(z, "manifest.json"))["frames"]
    for b in frame["boxes"]:
        assert [float(v) for v in lines[b["line"] - 1][1:]] == pytest.approx(b["bbox_xywh"])


def test_exif_timestamp_is_used_when_present(workspace: Path, tmp_path: Path):
    project = _project(workspace)
    img = tmp_path / "in" / "shot_0001.jpg"
    img.parent.mkdir(parents=True)
    exif = Image.Exif()
    exif[0x0132] = "2026:11:06 14:38:08"
    Image.new("RGB", (32, 24), (10, 20, 30)).save(img, exif=exif)
    imp = import_images(project, [img.parent])["import_id"]
    lid = _model_label_set(project, {}, imp)
    rec = export_label_set(project, lid)
    z = zipfile.ZipFile(project.store.root / rec["archive"])
    (f,) = json.loads(_read(z, "manifest.json"))["frames"]
    assert (f["timestamp"], f["timestamp_source"]) == ("2026:11:06 14:38:08", "exif")
    assert f["sequence"] == {"group": "shot", "index": 1, "source": "filename"}


def test_model_version_resolved_from_autolabel_weights(workspace: Path, tmp_path: Path):
    project = _project(workspace)
    _frames(tmp_path / "f", 1)
    imp = import_images(project, [tmp_path / "f"])["import_id"]
    lid = _model_label_set(project, {}, imp)
    project.store.update(
        "labels",
        lid,
        source={"kind": "model", "weights": "/x/artifacts/demo/runs/run-t1/out/best.pt"},
    )
    project.store.create(
        "models", "mv-007", {"model_version": "mv-007", "training_run_id": "run-t1"}
    )
    rec = export_label_set(project, lid)
    assert rec["auto_label"]["model_version"] == "mv-007"


def test_sequence_hint_patterns():
    assert sequence_hint("cam0_20261106_143807_t0031s") == {
        "group": "cam0_20261106_143807",
        "offset_s": 31,
        "source": "filename",
    }
    assert sequence_hint("frame_000002") == {"group": "frame", "index": 2, "source": "filename"}
    assert sequence_hint("IMG_0007")["group"] == "IMG"
    assert sequence_hint("12345") is None and sequence_hint("kitchen") is None


def test_windows_original_names_map_to_plain_file_names():
    """Archives zipped on Windows record ``images\\frame.jpg``; labels must be ``frame.txt``."""
    assert original_basename("images\\cam0_t0001s.jpg") == "cam0_t0001s.jpg"
    assert original_basename("a/b\\c.png") == "c.png" and original_basename("x.jpg") == "x.jpg"
    info = AssetInfo(
        asset_id="ab" * 32,
        ext=".jpg",
        width=1,
        height=1,
        size_bytes=1,
        format="JPEG",
        original_filenames=["images\\cam0_t0001s.jpg"],
        first_imported_at="2026-01-01T00:00:00Z",
    )
    assert info.stems == {"cam0_t0001s"}


def test_export_is_a_snapshot_and_never_touches_label_store(exported):
    project, lid, rec, z, _ = exported
    labels_dir = project.store.root / "labels"
    before = {p: p.read_bytes() for p in labels_dir.rglob("*") if p.is_file()}
    rec2 = export_label_set(project, lid)
    assert rec2["export_id"] != rec["export_id"]
    assert {p: p.read_bytes() for p in labels_dir.rglob("*") if p.is_file()} == before
    path = project.store.root / rec["archive"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == rec["sha256"]
    assert path.stat().st_size == rec["bytes"] and z.testzip() is None
    assert {n.split("/")[0] for n in z.namelist()} == {"demo_review"}
    for n in z.namelist():  # extractable on Windows unchanged
        assert not (set(n) & WINDOWS_BAD), n
    assert not list(path.parent.glob(".zip-*"))


def test_include_images_packs_original_file_names(exported):
    project, lid, _, _, _ = exported
    rec = export_label_set(project, lid, include_images=True)
    z = zipfile.ZipFile(project.store.root / rec["archive"])
    images = sorted(Path(n).name for n in z.namelist() if "/images/" in n)
    labels = sorted(Path(n).stem for n in z.namelist() if "/labels/" in n)
    assert [Path(i).stem for i in images] == labels and all(i.endswith(".jpg") for i in images)
    assert json.loads(z.read("demo_review/manifest.json"))["images_included"] is True


def test_invalid_ids_format_and_classes(exported):
    project, lid, _, _, _ = exported
    with pytest.raises(NotFoundError) as e:
        export_label_set(project, "lbl-20990101-000000-0000")
    assert e.value.code == "LABEL_NOT_FOUND"
    with pytest.raises(ConfigError) as e:
        export_label_set(project, "../labels/lbl-20260101-000000-abcd")
    assert e.value.code == "LABEL_SET_ID_INVALID"
    with pytest.raises(ConfigError) as e:
        export_label_set(project, lid, fmt="coco")
    assert e.value.code == "LABEL_FORMAT_UNSUPPORTED"
    text = re.sub(r"^\s*- dog\n|^\s*dog: dog\n", "", project.config_path.read_text(), flags=re.M)
    project.config_path.write_text(text)
    with pytest.raises(OperationError) as e:
        export_label_set(Project.load("demo", project.settings), lid)
    assert e.value.code == "LABEL_CLASS_UNKNOWN" and e.value.details["unknown_classes"] == ["dog"]


def _empty_label_set(project: Project, imp: str, lid: str) -> None:
    create_label_set(
        project,
        label_set_id=lid,
        annotations=[],
        source={"kind": "model"},
        scope_assets=scoped_asset_infos(project, [imp]),
        import_ids=[imp],
        labeled_asset_ids=set(),
    )


def test_ambiguous_file_names_refused(workspace: Path, tmp_path: Path):
    project = _project(workspace)
    make_image(tmp_path / "s" / "a" / "frame_1.jpg", seed=1)
    make_image(tmp_path / "s" / "b" / "frame_1.jpg", seed=2)
    imp = import_images(project, [tmp_path / "s"])["import_id"]
    _empty_label_set(project, imp, "lbl-20260101-000000-0001")
    with pytest.raises(OperationError) as e:
        export_label_set(project, "lbl-20260101-000000-0001")
    assert e.value.code == "AMBIGUOUS_STEMS"


def test_names_differing_only_in_case_refused(workspace: Path, tmp_path: Path):
    """Windows file names are case-insensitive: Frame_1.txt and frame_1.txt would collide."""
    project = _project(workspace)
    make_image(tmp_path / "s" / "Frame_1.jpg", seed=1)
    make_image(tmp_path / "s" / "frame_1.jpg", seed=2)
    imp = import_images(project, [tmp_path / "s"])["import_id"]
    _empty_label_set(project, imp, "lbl-20260101-000000-0002")
    with pytest.raises(OperationError) as e:
        export_label_set(project, "lbl-20260101-000000-0002")
    assert e.value.code == "AMBIGUOUS_STEMS"


def test_cli_export_exports_and_errors(exported):
    _, lid, rec, _, _ = exported
    r, p = invoke_json("label", "export", "demo", lid, "--format", "yolo")
    assert r.exit_code == 0, p
    res = p["result"]
    assert res["download"] == f"download demo {res['export_id']}"
    assert p["next_operations"][0] == res["download"]
    r, p = invoke_json("label", "exports", "demo")
    assert r.exit_code == 0 and len(p["result"]["exports"]) == 2
    assert {e["export_id"] for e in p["result"]["exports"]} >= {rec["export_id"]}
    r, p = invoke_json("label", "export", "demo", "lbl-20990101-000000-0000")
    assert r.exit_code == 3 and p["error"]["code"] == "LABEL_NOT_FOUND"
    r, p = invoke_json("label", "export", "demo", "../labels/x")
    assert r.exit_code == 2 and p["error"]["code"] == "LABEL_SET_ID_INVALID"
    r, p = invoke_json("label", "export", "nope", lid)
    assert r.exit_code != 0 and p["error"]["code"] == "PROJECT_NOT_FOUND"


def test_reviewed_labels_round_trip_with_lineage(exported, tmp_path: Path):
    project, lid, rec, z, imp = exported
    z.extractall(tmp_path / "win")
    review = tmp_path / "win" / "demo_review"
    (review / "labels" / "cam0_20261106_143807_t0002s.txt").write_text("1 0.5 0.5 0.1 0.1\n")
    new = import_labels(
        project,
        review / "labels",
        fmt="yolo",
        class_names=["cat", "dog"],
        import_ids=[imp],
        parent_label_set_id=lid,
    )
    assert new["parent_label_set_id"] == lid
    assert new["counts"]["annotations"] == 4 and new["counts"]["unmatched_label_files"] == 0
    assert new["counts"]["assets_without_annotations"] == 0  # empty files = reviewed negatives
    again = export_label_set(project, new["label_set_id"])
    m = json.loads(
        zipfile.ZipFile(project.store.root / again["archive"]).read("demo_review/manifest.json")
    )
    assert m["review"]["status"] == "imported" and m["auto_label"] is None
    assert m["label_set"]["parent_label_set_id"] == lid
    with pytest.raises(NotFoundError):
        import_labels(
            project, review / "labels", fmt="yolo", parent_label_set_id="lbl-20990101-000000-0000"
        )


# -- restricted download ---------------------------------------------------------------------------


def _download(workspace: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "AIP_REPO": str(workspace)}
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "download.py"), *args],
        capture_output=True,
        env=env,
        cwd=workspace,
        check=False,
    )


def test_download_streams_exact_bytes(exported, workspace: Path):
    project, _, rec, _, _ = exported
    raw = (project.store.root / rec["archive"]).read_bytes()
    p = _download(workspace, "demo", rec["export_id"])
    assert p.returncode == 0, p.stderr
    assert p.stdout == raw
    summary = json.loads(p.stderr)
    assert summary["status"] == "ok" and summary["result"]["sha256"] == rec["sha256"]

    p = _download(workspace, "demo", rec["export_id"], "--base64")
    alphabet = set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n")
    assert p.returncode == 0 and set(p.stdout) <= alphabet
    assert base64.b64decode(p.stdout) == raw
    # PowerShell text handling may turn LF into CRLF; the decoder must not care
    assert base64.b64decode(p.stdout.replace(b"\n", b"\r\n")) == raw

    p = _download(workspace, "demo", rec["export_id"], "--info")
    info = json.loads(p.stdout)
    assert p.returncode == 0 and info["result"]["bytes"] == len(raw)
    assert info["result"]["archive_name"] == rec["archive_name"]


@pytest.mark.parametrize(
    "args, code, exit_code",
    [
        ((), "USAGE", 2),
        (("demo",), "USAGE", 2),
        (("demo", "lex-20260101-000000-abcd", "--zip"), "USAGE", 2),
        (("Demo;rm", "lex-20260101-000000-abcd"), "PROJECT_NAME_INVALID", 2),
        (("demo", "../../etc/passwd"), "EXPORT_ID_INVALID", 2),
        (("demo", "lbl-20261106-075701-fa49"), "EXPORT_ID_INVALID", 2),
        (("other", "lex-20260101-000000-abcd"), "PROJECT_NOT_FOUND", 3),
        (("demo", "lex-20260101-000000-abcd"), "LABEL_EXPORT_NOT_FOUND", 3),
    ],
)
def test_download_refuses_anything_but_recorded_exports(exported, workspace, args, code, exit_code):
    p = _download(workspace, *args)
    assert p.returncode == exit_code and p.stdout == b""
    assert json.loads(p.stderr)["error"]["code"] == code


def test_download_refuses_tampered_or_redirected_records(exported, workspace: Path):
    project, _, rec, _, _ = exported
    path = project.store.root / rec["archive"]
    rec_path = project.store.root / "label_exports" / f"{rec['export_id']}.json"
    original = path.read_bytes()

    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    p = _download(workspace, "demo", rec["export_id"])
    assert p.returncode == 1 and p.stdout == b""
    assert json.loads(p.stderr)["error"]["code"] == "EXPORT_INTEGRITY_FAILED"
    path.write_bytes(original)

    secret = workspace / "infrastructure.yaml"
    for target in (os.path.relpath(secret, project.store.root), "labels/x.zip"):
        rec_path.write_text(json.dumps({**rec, "archive": target}))
        p = _download(workspace, "demo", rec["export_id"])
        assert p.returncode == 1 and p.stdout == b""
        assert json.loads(p.stderr)["error"]["code"] == "EXPORT_PATH_INVALID"


def test_wrapper_routes_download(exported, workspace: Path):
    """The forced-command path: SSH_ORIGINAL_COMMAND -> aip -> download.py."""
    _, _, rec, _, _ = exported
    (workspace / ".venv" / "bin").mkdir(parents=True)
    shim = workspace / ".venv" / "bin" / "python"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    shim.chmod(0o755)
    p = subprocess.run(
        ["bash", str(SCRIPTS / "aip")],
        capture_output=True,
        env={
            **os.environ,
            "AIP_REPO": str(workspace),
            "SSH_ORIGINAL_COMMAND": f"download demo {rec['export_id']} --info",
        },
        check=False,
    )
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["result"]["export_id"] == rec["export_id"]
