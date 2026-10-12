"""Nafas agent team: Scout, Explainer, Editor, DialectDesk, Auditor, Conductor.

Roles and handoffs
  Scout        reads the event stream and proposes moments (goal, save, press spell, chaos spell...)
  Explainer    turns a moment into structured, evidence-linked facts by calling the fact tools
  Editor       spends each viewer's attention budget (nafas.attention.schedule)
  DialectDesk  renders a scheduled line in the viewer's language and style
  Auditor      deterministic gate: numbers, names and wording must come from the facts
  Conductor    owns the shared state, hands work between agents, and recovers from failures

The language model is optional. Without credentials DialectDesk uses the deterministic phrasebook, so
the full pipeline runs offline. With `FOUNDRY_PROJECT_ENDPOINT` and `FOUNDRY_MODEL` set, DialectDesk asks a
Microsoft Agent Framework agent (Foundry chat client) to rephrase the line, and the Auditor still has the
last word. See README for what was and was not verified.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import phrasebook
from .attention import PROFILES, WORDS_PER_SEC, schedule
from .facts import MatchFacts, call_tool
from .simulator import MATCH_SECONDS, TEAMS, build_league, simulate_match, simulate_season
from .stats import GATE_DIRECTIONAL, Book, calibrate_goal_value

HISTORY_MATCHES = 38


def _mm(t: float) -> int:
    return int(t // 60) + 1


@dataclass
class Trace:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, t: float, agent: str, action: str, detail: str = "") -> None:
        self.entries.append({"t": round(t, 1), "agent": agent, "action": action, "detail": detail})


# ====================================================================================
# Scout
# ====================================================================================
class Scout:
    name = "Scout"

    def detect(self, facts: MatchFacts, trace: Trace) -> list[dict]:
        match = facts.match
        moments: list[dict] = []
        score = {"KBY": 0, "RMR": 0}
        press_times: dict[str, list[float]] = {t: [] for t in TEAMS}
        last_press: dict[str, float] = {t: -9999.0 for t in TEAMS}
        shot_times: dict[str, list[tuple[float, str, float]]] = {t: [] for t in TEAMS}
        last_chaos: dict[str, float] = {t: -9999.0 for t in TEAMS}
        shots = {s.id: s for s in match.shots}

        def mk(kind: str, t: float, imp: float, team: str, evidence: list[str], **raw: Any) -> None:
            moments.append({"id": f"MO{len(moments) + 1:03d}", "t": round(t, 1), "kind": kind,
                            "importance": round(min(imp, 1.0), 3), "team": team,
                            "evidence": evidence, "raw": raw})

        for e in match.events:
            typ, t = e["type"], e["t"]
            if typ == "shot":
                s = shots[e["shot_id"]]
                shot_times[s.team].append((t, s.id, s.xg))
                recent = [x for x in shot_times[s.team] if t - x[0] <= 150]
                if len(recent) >= 3 and t - last_chaos[s.team] > 220:
                    last_chaos[s.team] = t
                    mk("chaos", t, 0.45, s.team, [x[1] for x in recent], shots=len(recent),
                       xg_total=round(sum(x[2] for x in recent), 2))
            elif typ == "goal":
                s = shots[e["shot_id"]]
                score[s.team] += 1
                rarity = max(1, round(1 / s.xg))
                mk("goal", t, 0.85 + 0.15 * min(1.0, rarity / 30), s.team, [s.id, e["id"]],
                   shot_id=s.id, score=dict(score), rarity=rarity)
                mk("ledger", t + 0.5, 0.5, s.team, [s.id], shot_id=s.id, subject=s.shooter)
            elif typ == "save":
                s = shots[e["shot_id"]]
                mk("save", t, 0.35 + 0.5 * (s.psxg or 0), "RMR" if s.team == "KBY" else "KBY",
                   [s.id, e["id"]], shot_id=s.id)
                if (s.psxg or 0) >= 0.30:
                    mk("ledger", t + 0.5, 0.40, s.team, [s.id], shot_id=s.id, subject=s.keeper)
            elif typ == "off_target":
                s = shots[e["shot_id"]]
                if s.xg >= 0.05:
                    mk("missed", t, 0.18 + 0.7 * s.xg, s.team, [s.id], shot_id=s.id)
            elif typ == "pressure":
                press_times[e["team"]].append(t)
                recent = [x for x in press_times[e["team"]] if t - x <= 180]
                if len(recent) >= 6 and t - last_press[e["team"]] > 300:
                    last_press[e["team"]] = t
                    mk("press", t, 0.38, e["team"], [], count=len(recent))
            elif typ == "possession":
                mk("ambient", t, 0.10 + (0.04 if e.get("zone") == "attacking" else 0.0), e["team"],
                   [e["id"]], zone=e["zone"])
        for mark in (900, 1800, 2700, 3600, 4500, 5400):
            mk("state", mark, 0.50, "KBY", [], at=mark)
        moments.sort(key=lambda m: m["t"])
        for i, m in enumerate(moments, 1):
            m["id"] = f"MO{i:03d}"
        trace.add(0, self.name, "scan_complete",
                  f"{len(moments)} moments from {len(match.events)} events")
        return moments


# ====================================================================================
# Explainer
# ====================================================================================
class Explainer:
    name = "Explainer"

    def _club(self, team: str) -> dict:
        return TEAMS[team]

    def _pl(self, facts: MatchFacts, pid: str) -> dict:
        p = facts.players[pid]
        return {"id": pid, "name": p.name, "name_ar": p.name_ar}

    def explain(self, facts: MatchFacts, raw_moments: list[dict], trace: Trace) -> list[dict]:
        out: list[dict] = []
        for m in raw_moments:
            raw, kind, t = m["raw"], m["kind"], m["t"]
            f: dict[str, Any] = {"t": t}
            why: list[str] = []
            if kind in ("goal", "save", "missed"):
                shot = call_tool(facts, "get_shot", shot_id=raw["shot_id"])
                trace.add(t, self.name, "tool:get_shot", raw["shot_id"])
                f["shooter"] = self._pl(facts, shot["shooter"])
                f["keeper"] = self._pl(facts, shot["keeper"])
                f["side"] = shot["side"]
                f["xg"] = shot["xg"]
                if shot["psxg"] is not None:
                    f["psxg"] = shot["psxg"]
                if kind == "goal":
                    f["scorer"] = f["shooter"]
                    f["club"] = self._club(m["team"])
                    f["score"] = raw["score"]
                    f["rare"] = raw["rarity"] >= 15
                    why.append(f"chance quality {shot['xg']:.2f} xG, about 1 in {raw['rarity']}")
                    if shot["psxg"] is not None:
                        why.append(f"after the strike {round(shot['psxg'] * 100)}% to beat an average keeper")
                elif kind == "save":
                    f["big"] = (shot["psxg"] or 0) >= 0.45
                    why.append(f"post-shot xG {shot['psxg']:.2f}")
            elif kind == "press":
                f["club"] = self._club(m["team"])
                f["count"] = raw["count"]
                why.append(f"{raw['count']} pressure events within 180 s")
            elif kind == "chaos":
                f["club"] = self._club(m["team"])
                f["shots"], f["xg_total"] = raw["shots"], raw["xg_total"]
                why.append(f"{raw['shots']} shots in 150 s")
            elif kind == "state":
                st = call_tool(facts, "get_match_state", t=raw["at"])
                trace.add(t, self.name, "tool:get_match_state", f"t={raw['at']}")
                a, b = st["teams"]["KBY"], st["teams"]["RMR"]
                f.update({"minute": min(90, round(raw["at"] / 60)), "a": self._club("KBY"),
                          "b": self._club("RMR"), "ga": a["goals"], "gb": b["goals"],
                          "sa": a["shots"], "sb": b["shots"], "xa": a["xg"], "xb": b["xg"]})
            elif kind == "ambient":
                f["club"] = self._club(m["team"])
                f["zone"] = raw["zone"]
            elif kind == "ledger":
                pid = raw["subject"]
                after = call_tool(facts, "get_player_value", player_id=pid, t=t + 1.0)
                before = call_tool(facts, "get_player_value", player_id=pid, t=t, before_shot=raw["shot_id"])
                trace.add(t, self.name, "tool:get_player_value", pid)
                dec = after["decomposition"]
                f["player"] = self._pl(facts, pid)
                f["role"] = after["role"]
                f["n"] = after["n_shots"]
                f["gate"] = dec["gate"]
                f["need"] = max(0, GATE_DIRECTIONAL - after["n_shots"]) if dec["gate"] == "insufficient" else 0
                f["delta_observed"] = round(after["pulse_observed"] - before["pulse_observed"], 2)
                f["delta_evidence"] = round(after["pulse"]["mean"] - before["pulse"]["mean"], 2)
                f["luck_share"] = dec["luck_share"]["mean"]
                f["luck_lo"], f["luck_hi"] = dec["luck_share"]["lo"], dec["luck_share"]["hi"]
                why.append(f"evidence gate: {dec['gate']}")
                m["raw"]["row"] = {"pulse": after["pulse"], "pulse_observed": after["pulse_observed"],
                                   "luck_share": dec["luck_share"], "gate": dec["gate"]}
            out.append({**m, "facts": f, "why": why})
        facts.moments = out
        trace.add(0, self.name, "explained", f"{len(out)} moments with evidence links")
        return out


# ====================================================================================
# DialectDesk (+ optional Foundry backend)
# ====================================================================================
class Backend(Protocol):
    name: str

    async def rewrite(self, text: str, facts: dict, lang: str, style: str) -> str: ...


class TemplateBackend:
    name = "template"

    async def rewrite(self, text: str, facts: dict, lang: str, style: str) -> str:
        return text


class FoundryBackend:
    """Rephrases a templated line with a Microsoft Agent Framework agent on Foundry.

    Only constructed when FOUNDRY_PROJECT_ENDPOINT and FOUNDRY_MODEL are set. Requires
    `pip install agent-framework-foundry azure-identity` (pre-release) and `az login` or a managed identity.
    """

    name = "foundry"
    INSTRUCTIONS = (
        "You are the Dialect Desk of a football broadcast. Rewrite the given commentary line so it sounds "
        "natural for the requested language and viewer style. For Arabic use plain Saudi-flavoured wording. "
        "Hard rules: keep every number and every player or club name exactly as given; add no new facts, "
        "numbers or names; never give financial, betting or transfer advice; one or two short sentences."
    )

    def __init__(self) -> None:
        from agent_framework.foundry import FoundryChatClient  # type: ignore
        from azure.identity import DefaultAzureCredential  # type: ignore

        client = FoundryChatClient(
            project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
            model=os.environ["FOUNDRY_MODEL"],
            credential=DefaultAzureCredential(),
        )
        self.agent = client.as_agent(instructions=self.INSTRUCTIONS, name="dialect_desk")

    async def rewrite(self, text: str, facts: dict, lang: str, style: str) -> str:
        prompt = json.dumps({"language": lang, "style": style, "line": text}, ensure_ascii=False)
        result = await asyncio.wait_for(self.agent.run(prompt), timeout=8)
        return (getattr(result, "text", None) or str(result)).strip()


def make_backend() -> Backend:
    if os.environ.get("FOUNDRY_PROJECT_ENDPOINT") and os.environ.get("FOUNDRY_MODEL"):
        try:
            return FoundryBackend()
        except Exception as exc:  # missing package, auth problem...
            print(f"[nafas] Foundry backend unavailable ({exc}); using templates")
    return TemplateBackend()


class DialectDesk:
    name = "DialectDesk"

    def __init__(self, backend: Backend):
        self.backend = backend

    async def render(self, utt: dict) -> str:
        return await self.backend.rewrite(utt["text"], utt["facts"], utt["lang"], utt["style"])


# ====================================================================================
# Auditor (deterministic)
# ====================================================================================
_NUM = re.compile(r"\d+(?:\.\d+)?")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
BANNED = [
    "overpaid", "underpaid", "bargain", "rip-off", "guarantee", "will rise", "will score", "should buy",
    "should sell", "invest in", "buy ", "sell ", "اشتر", "بيع ", "مضمون", "استثمر", "ارتفاع السعر",
    "غالي", "رخيص", "راح يسجل", "ضمان",
]


def _collect_names(obj: Any, acc: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("name", "name_ar") and isinstance(v, str):
                acc.add(v)
            else:
                _collect_names(v, acc)


def _all_roster_names(facts: MatchFacts) -> set[str]:
    names = set()
    for p in facts.players.values():
        names |= {p.name, p.name_ar}
    for c in TEAMS.values():
        names |= {c["name"], c["name_ar"]}
    return names


class Auditor:
    """Numbers, names and wording in the final line must be derivable from the structured facts."""

    name = "Auditor"

    def __init__(self, facts: MatchFacts):
        self.roster = _all_roster_names(facts)

    def _allowed_numbers(self, utt: dict) -> set[str]:
        allowed: set[str] = set()
        for lang in ("en", "ar"):
            for style in ("analyst", "casual", "blind", "investor"):
                for level in ("full", "short"):
                    try:
                        txt = phrasebook.render(utt["kind"], utt["facts"], lang, style, level)
                    except Exception:
                        continue
                    allowed |= set(_NUM.findall(txt.translate(_AR_DIGITS)))
        return allowed

    def check(self, utt: dict, text: str) -> list[str]:
        problems: list[str] = []
        norm = text.translate(_AR_DIGITS)
        allowed = self._allowed_numbers(utt)
        for n in _NUM.findall(norm):
            if n not in allowed and n.lstrip("+-") not in allowed:
                problems.append(f"number not in facts: {n}")
        allowed_names: set[str] = set()
        _collect_names(utt["facts"], allowed_names)
        for name in self.roster:
            if name in text and name not in allowed_names:
                problems.append(f"name not in facts: {name}")
        low = text.lower()
        for b in BANNED:
            if b in low:
                problems.append(f"advice or certainty wording: {b.strip()}")
        if len(text.split()) > 2.5 * max(1, len(utt["text"].split())) + 6:
            problems.append("line much longer than the facts justify")
        return problems


# ====================================================================================
# Conductor
# ====================================================================================
class Conductor:
    name = "Conductor"

    def __init__(self, facts: MatchFacts, backend: Backend | None = None):
        self.facts = facts
        self.backend = backend or TemplateBackend()
        self.desk = DialectDesk(self.backend)
        self.auditor = Auditor(facts)
        self.trace = Trace()
        self.stats = {"rendered": 0, "fallbacks": 0, "blocked": 0, "backend_errors": 0}

    async def run_profile(self, moments: list[dict], profile: str, lang: str | None = None) -> dict:
        prof = PROFILES[profile]
        lang = lang or prof["lang"]
        plan = schedule(moments, profile, lang)
        for d in plan.dropped:
            self.trace.add(d["t"], "Editor", "dropped_stale", f"{profile}: {d['kind']} {d['moment_id']}")
        utterances: list[dict] = []
        for u in plan.utterances:
            if u["cut_off"]:
                self.trace.add(u["end"], "Editor", "cut_off",
                               f"{profile}: {u['kind']} cut by {u.get('cut_by')}")
            if u["catch_up"]:
                self.trace.add(u["start"], "Editor", "catch_up", f"{profile}: {u['kind']} compressed")
            text = u["text"]
            try:
                candidate = await self.desk.render(u)
            except Exception as exc:
                self.stats["backend_errors"] += 1
                self.stats["fallbacks"] += 1
                self.trace.add(u["start"], self.name, "fallback_to_template", f"backend error: {exc}")
                candidate = text
            problems = self.auditor.check(u, candidate)
            if problems:
                self.trace.add(u["start"], "Auditor", "blocked", f"{profile}: {problems[0]}")
                self.stats["fallbacks"] += 1
                candidate = text
                if self.auditor.check(u, candidate):
                    self.stats["blocked"] += 1
                    continue
            if u["catch_up"]:
                candidate = phrasebook.catch_up_prefix(lang) + candidate
            spoken = candidate
            if u["cut_off"]:
                words = candidate.split()
                keep = max(3, int(len(words) * (u["end"] - u["start"]) / max(u["dur"], 0.1)))
                spoken = " ".join(words[:keep]) + " ..."
            self.stats["rendered"] += 1
            utterances.append({
                "id": u["id"], "moment_id": u["moment_id"], "kind": u["kind"], "start": u["start"],
                "end": u["end"], "level": u["level"], "catch_up": u["catch_up"], "cut_off": u["cut_off"],
                "importance": u["importance"], "lang": lang, "style": u["style"],
                "text": spoken, "full_text": candidate, "evidence": u["evidence"],
            })
        metrics = dict(plan.metrics)
        metrics["audited_ok"] = len(utterances)
        return {"profile": profile, "label": prof["label"], "label_ar": prof["label_ar"], "lang": lang,
                "utterances": utterances, "dropped": plan.dropped, "metrics": metrics}


# ====================================================================================
# Session
# ====================================================================================
def build_facts(seed: int) -> MatchFacts:
    players = build_league(seed)
    history = simulate_season(seed, HISTORY_MATCHES)
    book = Book.from_history(players, history)
    calibrate_goal_value(book)
    return facts_for_match(seed, players, book, simulate_match(seed, HISTORY_MATCHES, players))


def facts_for_match(seed: int, players, book, match) -> MatchFacts:
    """Wire a match (simulated, or derived from video tracking) into the facts layer and live ledger."""
    facts = MatchFacts(seed=seed, match=match, players=players, book=book)
    book.set_live_fraction(0.0)
    facts.baseline_rows = {pid: book.row(pid) for pid in players}
    for s in sorted(match.shots, key=lambda x: x.t):
        outcome_t = s.t + 1.2
        book.set_live_fraction(outcome_t / MATCH_SECONDS)
        before = {pid: book.row(pid) for pid in (s.shooter, s.keeper)}
        book.add_live_shot(s)
        after = {pid: book.row(pid) for pid in (s.shooter, s.keeper)}
        facts.ledger_timeline.append({"t": round(outcome_t, 1), "shot_id": s.id, "rows": after,
                                      "before": before})
    return facts


async def build_session(seed: int = 7, backend: Backend | None = None, lang: str | None = None,
                        facts: MatchFacts | None = None) -> dict[str, Any]:
    facts = facts or build_facts(seed)
    conductor = Conductor(facts, backend)
    trace = conductor.trace
    raw = Scout().detect(facts, trace)
    moments = Explainer().explain(facts, raw, trace)
    profiles = {}
    for key in PROFILES:
        profiles[key] = await conductor.run_profile(moments, key, lang)
    # keep the trace compact and ordered
    trace.entries.sort(key=lambda e: e["t"])
    return {
        "meta": session_meta(facts, conductor),
        "events": [e for e in facts.match.events if e["type"] in
                   ("shot", "goal", "save", "off_target", "pressure", "possession")],
        "moments": [{k: v for k, v in m.items() if k != "raw"} for m in moments],
        "profiles": profiles,
        "ledger_final": facts.book.ledger(),
        "ledger_baseline": list(facts.baseline_rows.values()),
        "ledger_timeline": [{"t": s["t"], "shot_id": s["shot_id"], "rows": s["rows"],
                             "before": s["before"]} for s in facts.ledger_timeline],
        "trace": trace.entries[:400],
        "tool_calls": len(facts.tool_log),
        "conductor": conductor.stats,
    }


def session_meta(facts: MatchFacts, conductor: Conductor) -> dict[str, Any]:
    m = facts.match
    shots = m.shots
    hist_shots = [s for s in facts.book.shots if s.match != m.index]
    return {
        "seed": facts.seed, "teams": TEAMS, "duration": MATCH_SECONDS,
        "backend": conductor.backend.name,
        "players": [p.public() for p in facts.players.values()],
        "final_score": m.score,
        "match": {"shots": len(shots), "xg": round(sum(s.xg for s in shots), 2)},
        "history": {"matches": HISTORY_MATCHES, "shots": len(hist_shots)},
        "assumptions": {
            "synthetic_only": True,
            "finishing_skill_sd_logit": 0.20, "keeper_skill_sd_logit": 0.15,
            "evidence_gate_shots": {"directional": 100, "usable": 250},
            "goal_value_calibration": "median positive-value player is break-even (pulse 1.0)",
            "pulse_band": "covers finishing/keeping skill uncertainty only",
        },
        "profiles": {k: {"label": v["label"], "label_ar": v["label_ar"], "lang": v["lang"]}
                     for k, v in PROFILES.items()},
    }
