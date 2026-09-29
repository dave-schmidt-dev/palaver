"""CodexAdapter: event derivation, tool-call resolution, and tailing a rollout file."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from palaver.ingest.adapters.base import Adapter, Event, TailResult, read_complete_records
from palaver.ingest.cursors import Cursor

from .constants import (
    COMPACTED_RECORD_TYPE,
    COMPACTION_EVENT_TYPE,
    ERROR_EVENT_TYPE,
    EXEC_END_EVENT_TYPE,
    KIND_COMPACTION,
    KIND_ERROR,
    KIND_MESSAGE,
    KIND_SESSION_META,
    KIND_TURN_BOUNDARY,
    PATCH_END_EVENT_TYPE,
    STORE_GLOB,
    TURN_BOUNDARY_EVENT_TYPES,
)
from .ordering import order_records
from .records import CodexIdentity, _parse_record, _payload

# --- Adapter ----------------------------------------------------------------


def _event_kind(record: dict) -> str:
    """Map one record to its canonical event kind.

    Unrecognized record types fall through to the type's own name rather than
    being dropped, so a future Codex release's records are still ingested and
    still visible as evidence, just not specially interpreted.
    """
    rtype = record.get("type")

    if rtype == "session_meta":
        return KIND_SESSION_META
    if rtype == COMPACTED_RECORD_TYPE:
        return KIND_COMPACTION
    if rtype == "response_item":
        payload_type = _payload(record).get("type")
        if payload_type == "message":
            return KIND_MESSAGE
        return str(payload_type) if isinstance(payload_type, str) and payload_type else "unknown"
    if rtype == "event_msg":
        return _event_msg_kind(_payload(record))

    return str(rtype) if isinstance(rtype, str) and rtype else "unknown"


def _event_msg_kind(payload: dict) -> str:
    """Map an `event_msg` payload to its kind, checking all three error layers."""
    event_type = payload.get("type")

    if event_type in TURN_BOUNDARY_EVENT_TYPES:
        return KIND_TURN_BOUNDARY
    if event_type == COMPACTION_EVENT_TYPE:
        return KIND_COMPACTION
    if event_type == ERROR_EVENT_TYPE:
        return KIND_ERROR
    if event_type == EXEC_END_EVENT_TYPE and _exec_failed(payload):
        return KIND_ERROR
    if event_type == PATCH_END_EVENT_TYPE and payload.get("success") is False:
        return KIND_ERROR

    return str(event_type) if isinstance(event_type, str) and event_type else "unknown"


def _exec_failed(payload: dict) -> bool:
    """Whether an `exec_command_end` payload reports a failure.

    Both layers research §2 names are checked, independently: a non-zero
    `exit_code` and a `status` of `"failed"`. Requiring both would miss a
    release that stopped emitting one of them, and either alone is
    sufficient evidence of a failed command.
    """
    if payload.get("status") == "failed":
        return True
    exit_code = payload.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        return False
    return exit_code != 0


def _tool_call_id(record: dict) -> str | None:
    """Return the tool-call correlation id a record carries, if any."""
    payload = _payload(record)
    call_id = payload.get("call_id")
    return call_id if isinstance(call_id, str) and call_id else None


class CodexAdapter(Adapter):
    """Adapter over Codex's `~/.codex/sessions/**/rollout-*.jsonl` stores."""

    source = "codex"

    def __init__(self, root: Path | None = None) -> None:
        """Create the adapter.

        Args:
            root: Directory holding Codex's date-partitioned rollout files.
                Defaults to `~/.codex/sessions`; tests must always pass a
                `tmp_path` root instead (INV-2/INV-3) — this default is
                production-only and is never exercised by this module's own
                test suite.
        """
        self.root = Path(root) if root is not None else Path.home() / ".codex" / "sessions"

    def list_store_paths(self) -> Iterable[Path]:
        """Enumerate rollout files without opening any of them.

        Recursive, unlike `ClaudeCodeAdapter.list_store_paths`: Codex
        partitions its sessions root by `YYYY/MM/DD`, so the files are three
        levels down rather than one, and the depth carries no identity — the
        filename does. The `rollout-*` prefix keeps any other `.jsonl` a
        future release drops in that tree from being mistaken for a session
        store.
        """
        if not self.root.exists():
            return []
        return sorted(self.root.rglob(STORE_GLOB))

    def session_key_for(self, path: Path) -> str:
        """Derive session identity from the file path alone.

        A rollout filename embeds the thread's UUID, which
        `session_meta.payload.id` repeats, so the stem is a stable identity
        that costs no file read — and `discover_sessions` calls this for
        every path it enumerates, including ones it will never open.
        `read_identity` is the richer, file-reading counterpart for callers
        that need `cwd` or the parent-thread linkage.
        """
        return path.stem

    def project_key_for(self, path: Path) -> str | None:
        """Return the working directory this session ran in, or `None`.

        Unlike Claude Code, where the project is encoded in the containing
        directory name, Codex's date-partitioned layout carries no project
        information at all — `cwd` lives inside `session_meta`, so this must
        open the file.
        """
        identity = self.read_identity(path)
        return None if identity is None else identity.cwd

    def read_identity(self, path: Path) -> CodexIdentity | None:
        """Read `session_meta` identity out of a rollout file.

        Args:
            path: The rollout file.

        Returns:
            The session's identity, or `None` if the file carries no
            `session_meta` record — which a truncated or not-yet-flushed
            file legitimately may.
        """
        for record in self._records(path):
            if record.get("type") != "session_meta":
                continue
            payload = _payload(record)
            return CodexIdentity(
                cwd=_optional_str(payload.get("cwd")),
                id=_optional_str(payload.get("id")),
                session_id=_optional_str(payload.get("session_id")),
                parent_thread_id=_optional_str(payload.get("parent_thread_id")),
            )
        return None

    def has_unresolved_trailing_tool_use(self, path: Path) -> bool:
        """Report whether the session ended with a tool call still outstanding.

        Codex's turn boundary is the inversion that makes this different from
        Claude Code's version. There, the last line is bookkeeping and the
        check has to read *past* it to find the last real turn. Here, the
        last line usually *is* the boundary (296/300 sampled files), and a
        boundary means the turn is over — so `task_complete` and
        `turn_aborted` clear every pending call, and a file ending on one of
        them reports False no matter how many calls it opened along the way.
        Reading a dangling `function_call` as unresolved after its turn had
        already closed would pin a finished session into
        `discover_sessions`'s always-include path forever.

        Args:
            path: The rollout file.

        Returns:
            True when at least one tool call was opened, never answered, and
            never closed out by a turn boundary.
        """
        pending: set[str] = set()
        unkeyed_calls = 0
        for record in self._records(path):
            kind = _event_kind(record)
            if kind == KIND_TURN_BOUNDARY:
                pending.clear()
                unkeyed_calls = 0
                continue
            if kind == "function_call":
                call_id = _tool_call_id(record)
                if call_id is None:
                    # A call with no correlation id cannot be matched to its
                    # output, so it is counted rather than tracked. Ignoring
                    # it would under-report; the turn boundary still clears it.
                    unkeyed_calls += 1
                else:
                    pending.add(call_id)
                continue
            if kind == "function_call_output":
                call_id = _tool_call_id(record)
                if call_id is None:
                    unkeyed_calls = max(0, unkeyed_calls - 1)
                else:
                    pending.discard(call_id)
        return bool(pending) or unkeyed_calls > 0

    def tail(self, path: Path, cursor: Cursor) -> TailResult:
        """Read every complete record appended after `cursor` into canonical events.

        Args:
            path: The rollout file.
            cursor: The durable byte-offset cursor from the last tail.

        Returns:
            A `TailResult` whose events are in cursor order (`order_records`)
            and whose payloads are the decoded source records byte-for-byte,
            so the evidence INV-6 anchors into is the record as written, not
            a reshaped view of it.
        """
        raw_records, new_offset = read_complete_records(path, cursor.offset)
        session_key = self.session_key_for(path)
        records = []
        malformed_records = 0
        for raw in raw_records:
            record = _parse_record(raw, path)
            if record is not None:
                records.append(record)
            else:
                malformed_records += 1
        events = tuple(
            Event(session_key=session_key, kind=_event_kind(record), payload=record)
            for record in order_records(records)
        )
        return TailResult(
            events=events,
            cursor=Cursor(offset=new_offset),
            malformed_records=malformed_records,
        )

    def _records(self, path: Path) -> list[dict]:
        """Decode every complete record in `path`, in cursor order."""
        raw_records, _ = read_complete_records(path, 0)
        records = []
        for raw in raw_records:
            record = _parse_record(raw, path)
            if record is not None:
                records.append(record)
        return order_records(records)


def _optional_str(value: object) -> str | None:
    """Return `value` if it is a non-empty string, else `None`."""
    return value if isinstance(value, str) and value else None
