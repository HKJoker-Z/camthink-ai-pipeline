"""Group video-derived frames so a split cannot leak one recording into two partitions.

Label export already reads a frame's place in a clip from its file name
(``sequence_hint``). The dataset split uses the same hint:

- ``<clip>_t<seconds>s`` frames share a clip. Ordinary names (``img_001.jpg``) do not match
  and stay independent images.
- Clips named ``<camera>_<YYYYMMDD>_<HHMMSS>`` that follow each other on the same camera,
  with at most ``SESSION_GAP_S`` between one clip's last frame and the next clip's start,
  are one recording session. That is how consecutive five-minute files stay related.

A session is the leakage group. Callers decide whether a group may be cut on clip
boundaries; this module only identifies the groups.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ai_pipeline.core.data.assets import AssetInfo, original_basename

# Same patterns as the review-export manifest. Offset form is a video frame; the index form
# is recorded for reviewers but is also how ordinary numbered images are named, so the split
# does not treat it as a clip.
_SEQ_OFFSET = re.compile(r"^(?P<group>.+?)[_-]t(?P<offset>\d+(?:\.\d+)?)s$")
_SEQ_INDEX = re.compile(r"^(?P<group>.*?[A-Za-z].*?)[_-]?(?P<index>\d+)$")
_CLIP_CLOCK = re.compile(r"^(?P<camera>.+)_(?P<date>\d{8})_(?P<clock>\d{6})$")

# Silence longer than this, after a clip's last kept frame, starts a new recording session.
SESSION_GAP_S = 180


def sequence_hint(stem: str) -> dict[str, object] | None:
    """Parse ``<group>_t<seconds>s`` or ``<group>_<index>`` from a file stem.

    ``source`` is ``filename``: nothing is read from the image. ``None`` when the stem has
    neither shape (a bare number, or a name with no trailing index).
    """
    match = _SEQ_OFFSET.match(stem)
    if match:
        offset = float(match["offset"])
        return {
            "group": match["group"],
            "offset_s": int(offset) if offset.is_integer() else offset,
            "source": "filename",
        }
    match = _SEQ_INDEX.match(stem)
    if match:
        return {
            "group": match["group"].rstrip("_-"),
            "index": int(match["index"]),
            "source": "filename",
        }
    return None


@dataclass(frozen=True)
class Clip:
    clip_id: str
    asset_ids: tuple[str, ...]  # ascending offset, then asset id
    start_s: float | None  # seconds from a fixed epoch; None when the name has no clock
    end_s: float | None
    camera: str | None


@dataclass(frozen=True)
class LeakageGroup:
    """Frames that must not be shuffled across train / val / test."""

    group_id: str
    clips: tuple[Clip, ...]  # time order

    @property
    def asset_ids(self) -> tuple[str, ...]:
        return tuple(aid for clip in self.clips for aid in clip.asset_ids)


@dataclass(frozen=True)
class Grouping:
    by_asset: dict[str, str]  # asset id -> group id, video frames only
    groups: tuple[LeakageGroup, ...]

    @property
    def uses_video_groups(self) -> bool:
        return bool(self.groups)


def grouping_from_assets(assets: list[AssetInfo]) -> Grouping:
    """Clip and session groups for ``assets``. Images without a video stem are omitted."""
    frames: dict[str, list[tuple[float, str]]] = {}
    for asset in assets:
        parsed = _video_frame(asset)
        if parsed is None:
            continue
        clip_id, offset = parsed
        frames.setdefault(clip_id, []).append((offset, asset.asset_id))

    clips: list[Clip] = []
    for clip_id, items in frames.items():
        items.sort(key=lambda item: (item[0], item[1]))
        clock = _clip_clock(clip_id)
        start = end = None
        camera = None
        if clock is not None:
            camera, start = clock
            end = start + items[-1][0]
        clips.append(
            Clip(
                clip_id=clip_id,
                asset_ids=tuple(asset_id for _, asset_id in items),
                start_s=start,
                end_s=end,
                camera=camera,
            )
        )
    groups = tuple(
        LeakageGroup(group_id=_session_id(session), clips=tuple(session))
        for session in _cluster_sessions(clips)
    )
    by_asset = {aid: group.group_id for group in groups for aid in group.asset_ids}
    return Grouping(by_asset=by_asset, groups=groups)


def _video_frame(asset: AssetInfo) -> tuple[str, float] | None:
    """First original name (sorted) that is a ``_t<seconds>s`` frame, else None."""
    for name in sorted(asset.original_filenames):
        hint = sequence_hint(Path(original_basename(name)).stem)
        offset = None if hint is None else hint.get("offset_s")
        if hint is not None and isinstance(offset, (int, float)):
            return str(hint["group"]), float(offset)
    return None


def _clip_clock(clip_id: str) -> tuple[str, float] | None:
    match = _CLIP_CLOCK.match(clip_id)
    if not match:
        return None
    try:
        y, mo, d = int(match["date"][:4]), int(match["date"][4:6]), int(match["date"][6:8])
        hh, mm, ss = int(match["clock"][:2]), int(match["clock"][2:4]), int(match["clock"][4:6])
        stamp = datetime(y, mo, d, hh, mm, ss)  # names carry no timezone
    except ValueError:
        return None
    # Ordinal days keep a session that crosses midnight comparable without a timezone.
    start = stamp.toordinal() * 86400 + hh * 3600 + mm * 60 + ss
    return match["camera"], float(start)


def _cluster_sessions(clips: list[Clip]) -> list[list[Clip]]:
    timed = sorted(
        (c for c in clips if c.start_s is not None and c.camera),
        key=lambda c: (c.camera or "", c.start_s or 0, c.clip_id),
    )
    loose = sorted((c for c in clips if c.start_s is None or not c.camera), key=lambda c: c.clip_id)
    sessions: list[list[Clip]] = []
    for clip in timed:
        if (
            sessions
            and sessions[-1][-1].camera == clip.camera
            and clip.start_s is not None
            and sessions[-1][-1].end_s is not None
            and clip.start_s - sessions[-1][-1].end_s <= SESSION_GAP_S
        ):
            sessions[-1].append(clip)
        else:
            sessions.append([clip])
    sessions.extend([clip] for clip in loose)

    def sort_key(session: list[Clip]) -> tuple:
        first = session[0]
        return (first.start_s is None, first.start_s or 0, first.clip_id)

    sessions.sort(key=sort_key)
    return sessions


def _session_id(session: list[Clip]) -> str:
    first = min(session, key=lambda c: (c.start_s is None, c.start_s or 0, c.clip_id))
    if len(session) == 1:
        return f"clip:{first.clip_id}"
    return f"session:{first.clip_id}"
