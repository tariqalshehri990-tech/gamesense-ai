"""Match-facts MCP server.

Exposes the fact layer as MCP tools so any MCP client (Microsoft Agent Framework via
`MCPStreamableHTTPTool`, Claude, VS Code...) can query the same numbers the Nafas agents use.

Run:
    python -m nafas.mcp_server                      # stdio
    python -m nafas.mcp_server --http --port 8001   # streamable HTTP at /mcp
Environment: NAFAS_SEED (default 35)
"""
from __future__ import annotations

import argparse
import os

from mcp.server.fastmcp import FastMCP

from .agents import Explainer, Scout, Trace, build_facts
from .facts import MatchFacts, TOOL_SPECS

DEFAULT_SEED = int(os.environ.get("NAFAS_SEED", "35"))


def create_server(facts: MatchFacts | None = None, **settings) -> FastMCP:
    facts = facts or build_facts(DEFAULT_SEED)
    if not facts.moments:  # populate moments once so list_moments is useful
        trace = Trace()
        Explainer().explain(facts, Scout().detect(facts, trace), trace)
    mcp = FastMCP("nafas-match-facts", **settings)

    @mcp.tool(description=TOOL_SPECS["get_match_state"])
    def get_match_state(t: float = 5400.0) -> dict:
        return facts.get_match_state(t)

    @mcp.tool(description=TOOL_SPECS["get_shot"])
    def get_shot(shot_id: str) -> dict:
        return facts.get_shot(shot_id)

    @mcp.tool(description=TOOL_SPECS["get_player_value"])
    def get_player_value(player_id: str, t: float | None = None) -> dict:
        return facts.get_player_value(player_id, t)

    @mcp.tool(description=TOOL_SPECS["get_evidence"])
    def get_evidence(event_ids: list[str]) -> list[dict]:
        return facts.get_evidence(event_ids)

    @mcp.tool(description=TOOL_SPECS["list_moments"])
    def list_moments(min_importance: float = 0.0) -> list[dict]:
        return [{k: v for k, v in m.items() if k != "raw"} for m in facts.list_moments(min_importance)]

    @mcp.tool(description=TOOL_SPECS["whatif"])
    def whatif(shot_id: str, goal: bool) -> dict:
        return facts.whatif(shot_id, goal)

    return mcp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    if args.http:
        create_server(host=args.host, port=args.port).run(transport="streamable-http")
    else:
        create_server().run()


if __name__ == "__main__":
    main()
