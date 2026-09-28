#!/usr/bin/env python3
"""Deterministic stdio MCP server used by the commit regression harness."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

ECHO_TOOL = {
    "name": "echo",
    "description": "Return the supplied message",
    "inputSchema": {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("clean", "noisy", "silent"),
        default="clean",
        help="Server behavior used by a regression check",
    )
    return parser.parse_args()


def _write_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _flood_stderr() -> None:
    chunk = "diagnostic-" + ("x" * 4084)
    for _ in range(80):
        sys.stderr.write(chunk + "\n")
    sys.stderr.flush()


def _handle_request(request: dict[str, Any], mode: str) -> None:
    request_id = request.get("id")
    method = request.get("method")

    if mode == "silent":
        return

    if method == "initialize":
        if mode == "noisy":
            _flood_stderr()
        _write_json(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "local-regression-test", "version": "1.0.0"},
                },
            }
        )
        return

    if method == "tools/list":
        if mode == "noisy":
            _write_json(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/message",
                    "params": {"level": "info", "data": "still working"},
                }
            )
            _write_json(
                {
                    "jsonrpc": "2.0",
                    "id": 99999,
                    "result": {
                        "tools": [
                            {
                                "name": "stale_tool",
                                "description": "Must be ignored",
                                "inputSchema": {},
                            }
                        ]
                    },
                }
            )
            sys.stdout.write("INFO: non-JSON server log\n")
            sys.stdout.flush()

        _write_json(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": [ECHO_TOOL]},
            }
        )
        return

    if method == "tools/call":
        params = request.get("params", {})
        arguments = params.get("arguments", {})
        _write_json(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {"type": "text", "text": str(arguments.get("message", ""))}
                    ],
                    "isError": False,
                },
            }
        )
        return

    _write_json(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    )


def main() -> int:
    args = _parse_args()
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(request, dict):
            _handle_request(request, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
