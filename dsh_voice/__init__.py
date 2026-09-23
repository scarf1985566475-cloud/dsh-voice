"""dsh-voice: local speech transcription and English-to-Chinese live interpretation.

Three faces over one pipeline:

* :mod:`dsh_voice.server` — the loopback HTTP/WebSocket service the DSH panel talks to.
* :mod:`dsh_voice.mcp_server` — stdio MCP tools so the agent can transcribe files.
* :mod:`dsh_voice.cli` — the operator entry point (serve, transcribe, record, doctor).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
