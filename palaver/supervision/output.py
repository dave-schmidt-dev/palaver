"""Capture service output before application imports, using shared log rotation.

Only launchd uses this entry point. Ordinary CLI stdout remains untouched;
MCP uses HTTP. Separate files retain the streams, including partial lines.
"""

from __future__ import annotations

import codecs
import logging
import os
import selectors
import signal
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

from palaver.logging_setup import rotating_handler

CHUNK_BYTES = 4096


def _raise_log_error(_record):
    # A failed sink must stop the service, not silently discard its output.
    raise OSError("supervised output log write failed") from sys.exception()


def capture(command: list[str], stdout_path: Path, stderr_path: Path) -> int:
    """Drain both child streams in bounded chunks and return its exit status.

    Never buffer a whole line: tracebacks and newline-free floods are bounded
    too. Incremental decoding preserves UTF-8 split across reads; invalid
    bytes are replaced in these diagnostic files. SIGTERM/SIGINT reach the
    child, and launchd retains its default process-group cleanup on job exit.
    """
    with ExitStack() as stack:
        handlers = []
        for path in (stdout_path, stderr_path):
            handler = rotating_handler(path)
            stack.callback(handler.close)
            handlers.append(handler)
            handler.handleError = _raise_log_error
            handler.setFormatter(logging.Formatter("%(message)s"))
            handler.terminator = ""

        def emit(index: int, text: str) -> None:
            handler = handlers[index]
            # RotatingFileHandler counts characters; enforce bytes for UTF-8
            # too, and split supervisor diagnostics as well as child output.
            chunk_size = min(CHUNK_BYTES, handler.maxBytes // 4)
            for offset in range(0, len(text), chunk_size):
                chunk = text[offset:offset + chunk_size]
                if handler.stream.tell() + len(chunk.encode("utf-8")) > handler.maxBytes:
                    handler.doRollover()
                handler.handle(logging.LogRecord(
                    "palaver.supervision", logging.INFO, __file__, 0, chunk, (), None
                ))

        emit(1, "palaver supervision: starting service\n")
        try:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            emit(1, f"palaver supervision: cannot start service: {exc}\n")
            return 127

        def forward(signum, _frame):
            if child.poll() is None:
                child.send_signal(signum)

        for signum in (signal.SIGTERM, signal.SIGINT):
            previous = signal.signal(signum, forward)
            stack.callback(signal.signal, signum, previous)
        try:
            with selectors.DefaultSelector() as selector:
                decoders = [codecs.getincrementaldecoder("utf-8")("replace") for _ in handlers]
                for index, stream in enumerate((child.stdout, child.stderr)):
                    stack.callback(stream.close)
                    selector.register(stream, selectors.EVENT_READ, index)
                while selector.get_map():
                    ready = selector.select(timeout=15)
                    if not ready:
                        emit(1, "palaver supervision: service running; awaiting output\n")
                    for key, _ in ready:
                        data = os.read(key.fileobj.fileno(), CHUNK_BYTES)
                        emit(key.data, decoders[key.data].decode(data, final=not data))
                        if not data:
                            selector.unregister(key.fileobj)
            while True:
                try:
                    code = child.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    emit(1, "palaver supervision: service running; streams closed\n")
            emit(1, f"\npalaver supervision: service exited {code}\n")
            return code if code >= 0 else 128 - code
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


def main() -> int:
    """Run the internal capture command with two destinations and child argv."""
    if len(sys.argv) < 4:
        print("usage: output STDOUT_LOG STDERR_LOG COMMAND [ARG ...]", file=sys.stderr)
        return 2
    return capture(sys.argv[3:], Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    raise SystemExit(main())
