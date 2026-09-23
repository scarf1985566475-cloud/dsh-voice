"""Export meeting records into an Obsidian vault.

Two routes, tried in that order:

1. **Vault files.** Write the Markdown note straight into the vault directory.
   This is the default because it works whether or not Obsidian is running, and
   needs no plugin. Obsidian picks the file up on its next index.
2. **Local REST API.** When ``DSH_VOICE_OBSIDIAN_API_URL`` and an API key are
   configured, the note is PUT through the Local REST API plugin instead, which
   is how a running Obsidian instance learns about a change immediately.

The note is deliberately Obsidian-native rather than a dump: YAML frontmatter
for dataview queries, ``- [ ]`` checkboxes for action items so the tasks plugin
can collect them, and the full transcript inside a collapsed callout so a note
stays readable.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import load_settings
from .meetings import Meeting, MeetingError, MeetingStore

LOGGER = logging.getLogger("dsh_voice.obsidian")

#: Folder created inside the vault when none is configured.
DEFAULT_FOLDER = "80-会议记录"

#: Frontmatter tags every exported note carries.
DEFAULT_TAGS = ("会议记录", "dsh-voice")

#: Characters Obsidian cannot have in a filename.
_ILLEGAL_FILENAME = re.compile(r'[\\/:*?"<>|#^\[\]]')

#: The Obsidian application registry, used to find a vault when none is given.
_OBSIDIAN_REGISTRY = Path.home() / "Library" / "Application Support" / "obsidian" / "obsidian.json"


@dataclass
class ExportResult:
    """Where a note went and how it got there."""

    ok: bool
    path: str = ""
    vault: str = ""
    via: str = ""
    bytes: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize for the wire."""
        return {
            "ok": self.ok,
            "path": self.path,
            "vault": self.vault,
            "via": self.via,
            "bytes": self.bytes,
            **({"error": self.error} if self.error else {}),
        }


def list_vaults() -> list[Path]:
    """List Obsidian vaults registered with the desktop application.

    Returns:
        Vault directories, most recently opened first. Empty when the registry
        is missing or unreadable.
    """
    try:
        payload = json.loads(_OBSIDIAN_REGISTRY.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    vaults = payload.get("vaults")
    if not isinstance(vaults, dict):
        return []
    entries: list[tuple[int, Path]] = []
    for record in vaults.values():
        if not isinstance(record, dict):
            continue
        raw = record.get("path")
        if isinstance(raw, str) and raw:
            entries.append((int(record.get("ts") or 0), Path(raw)))
    entries.sort(key=lambda item: item[0], reverse=True)
    return [path for _ts, path in entries]


def resolve_vault(explicit: str | Path | None = None) -> Path:
    """Resolve which vault to export into.

    Args:
        explicit: Vault path from a caller; wins over everything else.

    Returns:
        The vault directory.

    Raises:
        MeetingError: When no vault can be determined.
    """
    if explicit:
        candidate = Path(explicit).expanduser()
        if not candidate.is_dir():
            raise MeetingError(f"vault not found: {candidate}")
        return candidate
    from_env = os.environ.get("DSH_VOICE_OBSIDIAN_VAULT")
    if from_env:
        candidate = Path(from_env).expanduser()
        if not candidate.is_dir():
            raise MeetingError(f"DSH_VOICE_OBSIDIAN_VAULT is not a directory: {candidate}")
        return candidate
    vaults = list_vaults()
    if vaults:
        return vaults[0]
    raise MeetingError(
        "no Obsidian vault found: pass --vault, set DSH_VOICE_OBSIDIAN_VAULT, "
        "or open a vault in Obsidian once so it registers itself",
    )


def note_filename(meeting: Meeting) -> str:
    """Build the note's filename from its date and title.

    Stable across re-exports on purpose: exporting the same meeting twice
    updates one note instead of littering the vault with copies.

    Args:
        meeting: The meeting to name.

    Returns:
        A sanitized ``YYYY-MM-DD 标题.md`` filename.
    """
    try:
        started = datetime.fromisoformat(meeting.meta.started_at)
        stamp = started.strftime("%Y-%m-%d")
    except ValueError:
        stamp = meeting.meta.id[:10]
    title = _ILLEGAL_FILENAME.sub("", meeting.meta.title).strip() or "会议"
    title = re.sub(r"\s+", " ", title)[:80]
    return f"{stamp} {title}.md"


def _yaml_string(value: str) -> str:
    """Quote a value for YAML frontmatter."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human string."""
    total = max(0, int(seconds))
    if total >= 3600:
        return f"{total // 3600}h{(total % 3600) // 60:02d}m"
    if total >= 60:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total}s"


def _clock(seconds: float) -> str:
    """Format a stream offset as ``MM:SS``."""
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def render_note(
    meeting: Meeting,
    *,
    tags: tuple[str, ...] = DEFAULT_TAGS,
    include_transcript: bool = True,
    transcript_collapsed: bool = True,
    source_label: str = "",
) -> str:
    """Render a meeting as an Obsidian note.

    Args:
        meeting: The meeting to render.
        tags: Frontmatter tags.
        include_transcript: Append the sentence-by-sentence record.
        transcript_collapsed: Put the transcript in a collapsed callout.
        source_label: Human label for where the audio came from.

    Returns:
        The complete Markdown note, frontmatter included.
    """
    meta = meeting.meta
    structured = meeting.summary_structured or {}
    title = structured.get("title") or meta.title
    duration = meta.wall_seconds or meta.duration_s
    origin = source_label or ("录音文件" if meta.source == "file" else "实时同传")

    lines: list[str] = ["---"]
    lines.append(f"title: {_yaml_string(title)}")
    lines.append(f"created: {meta.started_at}")
    lines.append(f"meeting_id: {_yaml_string(meta.id)}")
    lines.append(f"duration: {_yaml_string(_format_duration(duration))}")
    lines.append(f"segments: {meta.segments}")
    lines.append(f"source: {_yaml_string(origin)}")
    if meta.source_path:
        lines.append(f"audio: {_yaml_string(meta.source_path)}")
    lines.append("tags:")
    lines.extend(f"  - {tag}" for tag in tags)
    if structured.get("topics"):
        lines.append("topics:")
        lines.extend(f"  - {_yaml_string(str(topic))}" for topic in structured["topics"])
    lines.append("---")
    lines.append("")

    lines += [f"# {title}", ""]
    lines += [
        "> [!info] 会议信息",
        f"> 时间：{meta.started_at[:16].replace('T', ' ')} · 时长 {_format_duration(duration)} · "
        f"{meta.segments} 段 · 来源：{origin}",
        f"> 由 dsh-voice 本地识别（MLX Whisper）＋ DeepSeek 翻译/纪要保持同步",
        "",
    ]

    if meeting.summary:
        # Split into lines: the action-table rewrite below works line by line,
        # and a single blob would hide the table from it entirely.
        lines += _summary_body(meeting.summary).split("\n")
        lines.append("")
    else:
        overview = structured.get("overview")
        if overview:
            lines += ["## 概览", "", overview, ""]
        lines += ["> [!warning] 尚未生成纪要", "> 运行 `dsh-voice summarize " + meta.id + "` 生成。", ""]

    # Action items as real checkboxes: Obsidian's tasks plugin aggregates these
    # across the vault, which a Markdown table cannot participate in.
    actions = structured.get("action_items") or []
    if actions and "## 待办事项" in "\n".join(lines):
        lines = _replace_action_table(lines, actions)

    if include_transcript and meeting.segments:
        marker = "> [!quote]- 逐句记录" if transcript_collapsed else "> [!quote] 逐句记录"
        lines += [marker, f"> 共 {len(meeting.segments)} 句，英文为识别原文，中文为译文", ">"]
        for segment in meeting.segments:
            lines.append(f"> **[{_clock(segment.start)}]** {segment.en.strip()}")
            if segment.zh.strip():
                lines.append(f"> {segment.zh.strip()}")
            lines.append(">")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _summary_body(markdown: str) -> str:
    """Prepare stored minutes for embedding in a note.

    Drops the generated footer (the note carries its own metadata) and a leading
    H1, because the note already renders the title — leaving both would print
    the same heading twice.
    """
    text = markdown.strip()
    marker = "\n---\n\n*记录："
    if marker in text:
        text = text.split(marker)[0].rstrip()
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if line.lstrip().startswith("# "):
            del lines[index]
        break
    return "\n".join(lines).lstrip("\n").rstrip()


def _replace_action_table(lines: list[str], actions: list[dict[str, str]]) -> list[str]:
    """Rewrite the minutes' action table into Obsidian task checkboxes.

    Args:
        lines: The note's lines so far.
        actions: Canonical action items (``task``/``owner``/``due``).

    Returns:
        The lines with the table rows replaced.
    """
    result: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("| 事项 | 负责人 | 时间点 |"):
            # Skip the header, separator, and every data row.
            index += 2
            while index < len(lines) and lines[index].startswith("|"):
                index += 1
            for action in actions:
                bits = [action["task"]]
                if action.get("owner") and action["owner"] != "未提及":
                    bits.append(f"@{action['owner']}")
                if action.get("due") and action["due"] != "未提及":
                    bits.append(f"📅 {action['due']}")
                result.append(f"- [ ] {' — '.join(bits)}")
            continue
        result.append(line)
        index += 1
    return result


def _push_via_rest_api(note_path: str, markdown: str) -> ExportResult | None:
    """PUT a note through the Local REST API plugin, when it is configured.

    Args:
        note_path: Vault-relative note path.
        markdown: Note body.

    Returns:
        A result when the API was used, or ``None`` when it is not configured —
        the caller then falls back to writing the file directly.
    """
    base = os.environ.get("DSH_VOICE_OBSIDIAN_API_URL") or os.environ.get("OBSIDIAN_API_URL")
    if not base:
        return None
    key = os.environ.get("OBSIDIAN_API_KEY") or os.environ.get("DSH_VOICE_OBSIDIAN_API_KEY")
    if not key:
        LOGGER.warning("Obsidian REST API URL is set but no API key is; writing the vault file directly")
        return None
    import httpx

    url = f"{base.rstrip('/')}/vault/{note_path}"
    try:
        response = httpx.put(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "text/markdown"},
            content=markdown.encode("utf-8"),
            timeout=15.0,
        )
        response.raise_for_status()
    except Exception as error:  # noqa: BLE001 - fall back rather than lose the export
        LOGGER.warning("Obsidian REST API export failed (%s); falling back to a direct write", error)
        return None
    return ExportResult(ok=True, path=note_path, via="rest-api", bytes=len(markdown.encode("utf-8")))


def export_meeting(
    meeting_id: str,
    *,
    store: MeetingStore | None = None,
    vault: str | Path | None = None,
    folder: str | None = None,
    filename: str | None = None,
    tags: tuple[str, ...] = DEFAULT_TAGS,
    include_transcript: bool = True,
    overwrite: bool = True,
) -> ExportResult:
    """Export one recorded meeting into an Obsidian vault.

    Args:
        meeting_id: Meeting to export.
        store: Meeting store; the configured one when omitted.
        vault: Vault directory; resolved from the config or registry when omitted.
        folder: Folder inside the vault; ``DSH_VOICE_OBSIDIAN_FOLDER`` or
            :data:`DEFAULT_FOLDER` when omitted.
        filename: Explicit filename, used instead of the derived one.
        tags: Frontmatter tags.
        include_transcript: Append the sentence-by-sentence record.
        overwrite: Replace an existing note rather than refusing.

    Returns:
        The export result; failures are reported as data, not raised, so a tool
        caller can see what happened.

    Raises:
        MeetingError: When the meeting or the vault cannot be resolved.
    """
    settings = load_settings()
    store = store or MeetingStore(settings.meetings_dir)
    meeting = store.read(meeting_id)
    target_folder = folder or os.environ.get("DSH_VOICE_OBSIDIAN_FOLDER") or DEFAULT_FOLDER
    vault_dir = resolve_vault(vault)

    markdown = render_note(
        meeting,
        tags=tags,
        include_transcript=include_transcript,
        source_label="录音文件" if meeting.meta.source == "file" else "实时同传",
    )
    relative = str(Path(target_folder) / (filename or note_filename(meeting))) if target_folder else (
        filename or note_filename(meeting)
    )

    pushed = _push_via_rest_api(relative, markdown)
    if pushed is not None:
        pushed.vault = str(vault_dir)
        return pushed

    destination = vault_dir / relative
    if destination.exists() and not overwrite:
        return ExportResult(
            ok=False, vault=str(vault_dir), path=str(destination),
            error=f"note already exists: {destination}",
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(markdown, encoding="utf-8")
    except OSError as error:
        return ExportResult(ok=False, vault=str(vault_dir), path=str(destination), error=str(error))
    return ExportResult(
        ok=True, path=str(destination), vault=str(vault_dir), via="vault-file",
        bytes=len(markdown.encode("utf-8")),
    )
