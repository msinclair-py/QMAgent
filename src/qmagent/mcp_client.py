"""Small CLI client for the QMAgent MCP server.

This is intentionally thin: it proves and exercises the package plumbing. The
server owns the persistent Academy QMAgent; this client only submits tool calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def _content_to_json(result: Any) -> Any:
    """Decode a FastMCP tool result into Python data when possible."""
    content = getattr(result, "content", result)
    if isinstance(content, list) and content:
        text = getattr(content[0], "text", None)
        if text is not None:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return content


async def call_tool(url: str, name: str, arguments: dict[str, Any],
                    timeout: float = 30.0) -> Any:
    async with streamablehttp_client(
        url,
        timeout=timeout,
        sse_read_timeout=timeout,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            return _content_to_json(result)


async def amain() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765/mcp")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="HTTP/SSE timeout in seconds for the MCP tool call")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health")

    p = sub.add_parser("run-parameterization")
    p.add_argument("--smiles", required=True)
    p.add_argument("--resname", required=True)
    p.add_argument("--charge", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--basis", default="def2-svp")
    p.add_argument("--functional", default="b3lyp")
    p.add_argument("--dispersion", default="d3bj")
    p.add_argument("--grid-level", type=int, default=3)
    p.add_argument("--scan-step", type=int, default=15)

    p = sub.add_parser("execute-code")
    p.add_argument("--code", help="Python code to execute through QMAgent")
    p.add_argument("--code-file", help="File containing Python code to execute")
    p.add_argument("--workdir", help="Working directory on the QMAgent side")
    p.add_argument("--exec-timeout", type=float, default=1800.0,
                   help="Timeout passed to QMAgent.execute_code")

    args = parser.parse_args()
    if args.cmd == "health":
        payload = await call_tool(args.url, "health", {}, timeout=args.timeout)
    elif args.cmd == "run-parameterization":
        payload = await call_tool(args.url, "run_parameterization", {
            "smiles": args.smiles,
            "resname": args.resname,
            "charge": args.charge,
            "output_dir": args.output_dir,
            "basis": args.basis,
            "functional": args.functional,
            "dispersion": args.dispersion,
            "grid_level": args.grid_level,
            "scan_step": args.scan_step,
        }, timeout=args.timeout)
    elif args.cmd == "execute-code":
        if bool(args.code) == bool(args.code_file):
            parser.error("execute-code requires exactly one of --code or --code-file")
        code = args.code if args.code is not None else open(args.code_file).read()
        payload = await call_tool(args.url, "execute_code", {
            "code": code,
            "workdir": args.workdir,
            "timeout": args.exec_timeout,
        }, timeout=args.timeout)
    else:  # pragma: no cover - argparse enforces choices
        raise AssertionError(args.cmd)

    print(json.dumps(payload, indent=2, default=str))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
