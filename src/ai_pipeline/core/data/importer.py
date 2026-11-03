"""``data import``: copy source images into the asset store and record the import event."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_pipeline.core.config.schema import LocalImagesSource
from ai_pipeline.core.data.assets import IMAGE_EXTENSIONS, AssetStore
from ai_pipeline.core.errors import ConfigError, OperationError
from ai_pipeline.core.ids import new_event_id, utc_now_iso
from ai_pipeline.core.logging import get_logger
from ai_pipeline.core.project import Project

log = get_logger("data.import")


def asset_store(project: Project) -> AssetStore:
    return AssetStore(project.artifact_dir / "assets")


def resolve_sources(project: Project, source: Path | None) -> list[Path]:
    """Explicit ``source`` wins; otherwise every ``local_images`` source from project.yaml,
    resolved relative to the project directory."""
    if source is not None:
        return [source.expanduser().resolve()]
    paths = [
        (project.dir / s.path).resolve()
        for s in project.config.dataset.sources
        if isinstance(s, LocalImagesSource)
    ]
    if not paths:
        raise ConfigError(
            f"Project {project.name!r} has no local_images source and no <source> was given",
            code="NO_LOCAL_IMAGE_SOURCE",
            hint="Pass a path: `ai-pipeline data import <project> <dir-or-file>`, "
            "or add a `type: local_images` entry to dataset.sources.",
        )
    return paths


def _iter_files(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root]
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in it if p.is_file())


def import_images(
    project: Project, sources: list[Path], *, recursive: bool = True
) -> dict[str, Any]:
    """Import every image under ``sources``. Idempotent on content: identical bytes are
    recognised and not stored twice. Always records an import event."""
    for src in sources:
        if not src.exists():
            raise OperationError(
                f"Import source does not exist: {src}",
                code="IMPORT_SOURCE_MISSING",
                details={"source": str(src)},
                hint="Check the path, or edit dataset.sources[].path in project.yaml.",
            )

    import_id = new_event_id("imp")
    store = asset_store(project)
    blog = log.bind(project=project.name, operation="data.import")
    assets: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    skipped_non_image = 0
    files_seen = 0

    for src in sources:
        for path in _iter_files(src, recursive):
            files_seen += 1
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                skipped_non_image += 1
                continue
            rel = str(path.relative_to(src)) if src.is_dir() else path.name
            try:
                info, created = store.add(path, original_filename=rel, import_id=import_id)
            except ValueError as exc:
                rejected.append({"path": str(path), "reason": str(exc)})
                blog.warning("rejected %s: %s", path, exc)
                continue
            assets.append({"asset_id": info.asset_id, "filename": rel, "new": created})

    if not assets and not rejected:
        raise OperationError(
            "No image files found in " + ", ".join(str(s) for s in sources),
            code="NO_IMAGES_FOUND",
            details={
                "sources": [str(s) for s in sources],
                "files_seen": files_seen,
                "supported_extensions": sorted(IMAGE_EXTENSIONS),
            },
            hint="Place images under the source directory or pass a different path.",
        )

    new_count = sum(1 for a in assets if a["new"])
    record = {
        "import_id": import_id,
        "state": "IMPORTED",
        "created_at": utc_now_iso(),
        "sources": [str(s) for s in sources],
        "recursive": recursive,
        "counts": {
            "files_seen": files_seen,
            "images_new": new_count,
            "images_existing": len(assets) - new_count,
            "rejected": len(rejected),
            "skipped_non_image": skipped_non_image,
        },
        "assets": assets,
        "rejected": rejected,
    }
    project.store.create("imports", import_id, record)
    blog.info(
        "import %s: %d new, %d existing, %d rejected",
        import_id,
        new_count,
        len(assets) - new_count,
        len(rejected),
    )
    return record


def scoped_asset_infos(project: Project, import_ids: list[str] | None):
    """Assets belonging to the given imports (default: all assets in the store)."""
    store = asset_store(project)
    infos = store.all_infos()
    if import_ids:
        for iid in import_ids:
            project.store.read("imports", iid)  # raises NotFoundError with hint
        wanted = set(import_ids)
        infos = [a for a in infos if wanted & set(a.import_ids)]
    return infos
