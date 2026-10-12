import asyncio
import statistics

from nafas import attention, simulator, stats, xg
from nafas.agents import Auditor, Conductor, Explainer, Scout, Trace, build_facts, build_session, make_backend


def test_xg_monotonic():
    near = xg.ShotFeatures(distance=8, angle=0.9)
    far = xg.ShotFeatures(distance=28, angle=0.3)
    assert xg.xg(near) > xg.xg(far) > 0


def test_simulator_deterministic_and_calibrated():
    a = simulator.simulate_season(7, 38)
    b = simulator.simulate_season(7, 38)
    assert [len(m.shots) for m in a] == [len(m.shots) for m in b]
    per_match = sum(len(m.shots) for m in a) / len(a)
    assert 18 < per_match < 30  # Premier-League-like volume


def test_evidence_gate_and_verdict_codes(facts):
    rows = facts.book.ledger()
    assert rows
    for r in rows:
        if r["n_shots"] < stats.GATE_DIRECTIONAL:
            assert r["verdict"]["code"] == "no_evidence"
            assert r["decomposition"]["gate"] == "insufficient"
        lo, hi = r["pulse"]["lo"], r["pulse"]["hi"]
        assert lo <= hi


def test_hidden_skill_is_recovered_across_seasons():
    """Finishers with truly positive skill should on average get a higher posterior mean than negative ones."""
    pos, neg = [], []
    for seed in (1, 2, 3):
        f = build_facts(seed)
        for r in f.book.ledger():
            p = f.players[r["id"]]
            if r["role"] in ("GK",) or r["decomposition"]["gate"] == "insufficient":
                continue
            (pos if p.finishing > 0 else neg).append(r["decomposition"]["theta"]["mean"])
    assert pos and neg
    assert statistics.mean(pos) > statistics.mean(neg)


def test_session_all_profiles_silence_and_cutoffs(facts):
    s = asyncio.run(build_session(35, backend=make_backend(), lang=None, facts=facts))
    assert set(s["profiles"]) == set(attention.PROFILES)
    assert sum(p["metrics"]["cut_offs"] for p in s["profiles"].values()) >= 1
    for p in s["profiles"].values():
        assert p["metrics"]["silence_share"] > 0.5


def test_scheduler_is_deterministic(facts):
    t = Trace()
    moments = Explainer().explain(facts, Scout().detect(facts, t), t)
    a = attention.schedule(moments, "casual")
    b = attention.schedule(moments, "casual")
    assert [u["text"] for u in a.utterances] == [u["text"] for u in b.utterances]


def test_auditor_blocks_advice_and_unknown_numbers(facts):
    t = Trace()
    moments = Explainer().explain(facts, Scout().detect(facts, t), t)
    m = next(x for x in moments if x["kind"] == "goal")
    utt = {"kind": m["kind"], "facts": m["facts"], "text": "x y z", "lang": "en", "style": "analyst"}
    aud = Auditor(facts)
    assert any("advice" in p for p in aud.check(utt, "He will score again, you should buy"))
    assert any("number" in p for p in aud.check(utt, "A strike worth 98765 goals"))
