"""The Editor: schedules commentary against each viewer's attention budget.

Idea: a viewer's attention is a budget, not a bucket to fill. Each profile has a bucket of
"speech seconds" that refills over match time. The Editor decides, in match time:
  * what is worth saying at all (profile weights x importance vs a threshold)
  * what to say first when several moments compete
  * when to stay silent (bucket empty, nothing above threshold)
  * when to cut a sentence off because a bigger moment arrived (preemption)
  * when a queued item has gone stale (compress to a short catch-up line, or drop it)

Everything is computed in event time, so the result is deterministic and testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import phrasebook
from .simulator import MATCH_SECONDS

WORDS_PER_SEC = {"en": 2.7, "ar": 2.3}

PROFILES: dict[str, dict[str, Any]] = {
    "analyst": {
        "label": "Analyst", "label_ar": "محلل", "lang": "en", "style": "analyst",
        "refill": 0.50, "capacity": 18.0, "ttl": 25.0, "min_importance": 0.15,
        "weights": {"goal": 1.0, "save": 1.0, "missed": 1.0, "press": 1.0, "chaos": 1.0,
                    "ledger": 0.0, "ambient": 0.0, "state": 1.0},
    },
    "casual": {
        "label": "Casual fan", "label_ar": "مشجع عادي", "lang": "ar", "style": "casual",
        "refill": 0.28, "capacity": 9.0, "ttl": 14.0, "min_importance": 0.30,
        "weights": {"goal": 1.0, "save": 1.0, "missed": 0.9, "press": 0.7, "chaos": 0.9,
                    "ledger": 0.0, "ambient": 0.0, "state": 0.8},
    },
    "blind": {
        "label": "Audio description", "label_ar": "وصف صوتي", "lang": "ar", "style": "blind",
        "refill": 0.62, "capacity": 20.0, "ttl": 18.0, "min_importance": 0.08,
        "weights": {"goal": 1.0, "save": 1.0, "missed": 1.0, "press": 0.9, "chaos": 1.0,
                    "ledger": 0.0, "ambient": 1.0, "state": 0.9},
    },
    "investor": {
        "label": "Investor", "label_ar": "مستثمر", "lang": "ar", "style": "investor",
        "refill": 0.26, "capacity": 12.0, "ttl": 40.0, "min_importance": 0.20,
        "weights": {"goal": 0.8, "save": 0.7, "missed": 0.0, "press": 0.0, "chaos": 0.0,
                    "ledger": 1.9, "ambient": 0.0, "state": 0.0},
    },
}


def _estimate(m: dict, prof: dict, lang: str, level: str) -> tuple[str, float]:
    text = phrasebook.render(m["kind"], m["facts"], lang, prof["style"], level)
    return text, max(1.5, len(text.split()) / WORDS_PER_SEC[lang])


@dataclass
class Result:
    utterances: list[dict]
    dropped: list[dict]
    filtered: int
    metrics: dict[str, Any]


def schedule(moments: list[dict], profile_key: str, lang: str | None = None) -> Result:
    prof = PROFILES[profile_key]
    lang = lang or prof["lang"]
    refill, cap, ttl = prof["refill"], prof["capacity"], prof["ttl"]

    cands: list[dict] = []
    filtered = 0
    for m in sorted(moments, key=lambda x: x["t"]):
        imp = m["importance"] * prof["weights"].get(m["kind"], 0.0)
        if imp < prof["min_importance"]:
            filtered += 1
            continue
        cands.append({**m, "imp": imp, "level": "full", "catch_up": False})

    state = {"bucket": cap, "last": 0.0, "now": 0.0, "cur": None}
    queue: list[dict] = []
    out: list[dict] = []
    dropped: list[dict] = []

    def do_refill(t: float) -> None:
        t = max(t, state["last"])
        state["bucket"] = min(cap, state["bucket"] + refill * (t - state["last"]))
        state["last"] = t

    def finish_cur() -> None:
        cur = state["cur"]
        out.append(cur)
        state["now"] = cur["end"]
        state["cur"] = None

    def run_until(T: float) -> None:
        while True:
            cur = state["cur"]
            if cur is not None:
                if cur["end"] <= T:
                    finish_cur()
                else:
                    return
            if not queue:
                return
            earliest = min(q["t"] for q in queue)
            start_t = max(state["now"], earliest)
            avail = [q for q in queue if q["t"] <= start_t]
            # staleness at the moment we could start
            stale = [q for q in avail if start_t - q["t"] > ttl * (2 if q["catch_up"] else 1)]
            if stale:
                q = stale[0]
                if q["imp"] >= 0.55 and not q["catch_up"]:
                    q["level"], q["catch_up"] = "short", True
                else:
                    queue.remove(q)
                    dropped.append({"moment_id": q["id"], "kind": q["kind"], "t": q["t"],
                                    "reason": "stale"})
                continue
            best = max(avail, key=lambda q: (q["imp"], -q["t"]))
            text, dur = _estimate(best, prof, lang, best["level"])
            do_refill(start_t)
            need = dur * 0.6
            if state["bucket"] < need:
                start_t += (need - state["bucket"]) / refill
                if start_t > T:
                    return
                do_refill(start_t)
                state["now"] = start_t  # the clock really advanced while we waited for budget
                if start_t - best["t"] > ttl * (2 if best["catch_up"] else 1):
                    continue  # went stale while waiting for budget; re-evaluate
            if start_t > T:
                return
            state["bucket"] -= dur
            queue.remove(best)
            state["cur"] = {
                "moment_id": best["id"], "kind": best["kind"], "t_moment": best["t"],
                "start": round(start_t, 1), "end": round(start_t + dur, 1), "dur": dur,
                "level": best["level"], "catch_up": best["catch_up"], "cut_off": False,
                "importance": round(best["imp"], 2), "lang": lang, "style": prof["style"],
                "text": text, "facts": best["facts"], "evidence": best["evidence"],
            }

    for c in cands:
        run_until(c["t"])
        cur = state["cur"]
        if cur is not None and c["imp"] >= cur["importance"] + 0.25:
            remaining = (cur["end"] - c["t"]) / max(cur["dur"], 0.1)
            spoken = c["t"] - cur["start"]
            # only cut a sentence that is well under way and still has plenty left to say
            if remaining > 0.35 and spoken >= 2.0:
                state["bucket"] += max(0.0, cur["dur"] - spoken)  # refund the unspoken part
                cur["end"] = round(c["t"], 1)
                cur["cut_off"] = True
                cur["cut_by"] = c["id"]
                finish_cur()
        queue.append(c)
        run_until(c["t"])
    run_until(MATCH_SECONDS * 10)
    if state["cur"] is not None:
        finish_cur()

    out.sort(key=lambda u: u["start"])
    for i, u in enumerate(out, 1):
        u["id"] = f"U-{profile_key[:3]}-{i:03d}"
    spoken = sum(u["end"] - u["start"] for u in out)
    metrics = {
        "utterances": len(out),
        "speech_seconds": round(spoken, 1),
        "silence_seconds": round(max(0.0, MATCH_SECONDS - spoken), 1),
        "silence_share": round(max(0.0, 1 - spoken / MATCH_SECONDS), 3),
        "cut_offs": sum(1 for u in out if u["cut_off"]),
        "catch_ups": sum(1 for u in out if u["catch_up"]),
        "dropped_stale": len(dropped),
        "filtered_low_value": filtered,
        "candidates": len(cands),
    }
    return Result(out, dropped, filtered, metrics)
