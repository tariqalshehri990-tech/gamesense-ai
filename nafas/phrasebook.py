"""Deterministic commentary templates (English and Saudi-flavoured Arabic).

These are both the offline fallback and the cost estimator for the attention scheduler.
Arabic is written in plain Gulf-friendly wording; the voice used by Azure AI Speech is a Saudi-locale
Modern Standard Arabic voice, so we describe the text as "Saudi-flavoured" and the audio as
"Saudi-locale". Numbers are written with Western digits in both languages so they can be audited.
"""
from __future__ import annotations

from typing import Any

SIDE_AR = {"left": "الجهة اليسرى", "right": "الجهة اليمنى", "centre": "من الوسط"}
SIDE_EN = {"left": "from the left", "right": "from the right", "centre": "down the middle"}


def _mm(t: float) -> int:
    return int(t // 60) + 1


def _pct(x: float) -> int:
    return int(round(x * 100))


def _n(f: dict, key: str, lang: str) -> str:
    return f[key]["name_ar" if lang == "ar" else "name"]


def _club(f: dict, key: str, lang: str) -> str:
    return f[key]["name_ar" if lang == "ar" else "name"]


# ---- kinds -------------------------------------------------------------------------------
def goal(f: dict, lang: str, style: str, level: str) -> str:
    m, xg = _mm(f["t"]), f["xg"]
    one_in = max(1, round(1 / xg)) if xg > 0 else 99
    sc = f["score"]
    score = f"{sc['KBY']}-{sc['RMR']}"
    who, club = _n(f, "scorer", lang), _club(f, "club", lang)
    if lang == "ar":
        if level == "short":
            return f"هدف ل{club}! {who}، الدقيقة {m}. النتيجة {score}."
        base = f"هدف! {who} يسجل ل{club} في الدقيقة {m}. النتيجة {score}."
        if style == "analyst":
            base += f" الفرصة كانت بقيمة {xg:.2f} هدف متوقع، واحدة من كل {one_in} تنتهي هدف."
        elif style == "blind":
            base += f" التسديدة جات {SIDE_AR[f['side']]}، وانتهت داخل الشباك."
        elif f["rare"]:
            base += f" فرصة نادرة، واحدة من كل {one_in}."
        return base
    if level == "short":
        return f"Goal for {club}! {who}, minute {m}. It is {score}."
    base = f"Goal! {who} scores for {club} in minute {m}. It is {score}."
    if style == "analyst":
        base += f" The chance was worth {xg:.2f} expected goals, about one in {one_in} goes in."
    elif style == "blind":
        base += f" The shot came {SIDE_EN[f['side']]} and ended in the net."
    elif f["rare"]:
        base += f" A rare finish, about one in {one_in}."
    return base


def save(f: dict, lang: str, style: str, level: str) -> str:
    m, ps = _mm(f["t"]), f["psxg"]
    k, s = _n(f, "keeper", lang), _n(f, "shooter", lang)
    if lang == "ar":
        if level == "short":
            return f"تصدي من {k} في الدقيقة {m}."
        base = f"تصدي من {k}! تسديدة {s} في الدقيقة {m}."
        if style == "analyst":
            base += f" احتمال الهدف بعد التسديدة كان {_pct(ps)} بالمية."
        elif style == "blind":
            base += f" التسديدة جات {SIDE_AR[f['side']]}."
        elif f["big"]:
            base += " تصدي كبير."
        return base
    if level == "short":
        return f"Save by {k}, minute {m}."
    base = f"Save by {k}! {s} shoots in minute {m}."
    if style == "analyst":
        base += f" The shot had a {_pct(ps)} percent chance of beating an average keeper."
    elif style == "blind":
        base += f" The shot came {SIDE_EN[f['side']]}."
    elif f["big"]:
        base += " A big save."
    return base


def missed(f: dict, lang: str, style: str, level: str) -> str:
    m, xg = _mm(f["t"]), f["xg"]
    s = _n(f, "shooter", lang)
    if lang == "ar":
        base = f"فرصة ضايعة! {s} يسدد برا المرمى في الدقيقة {m}."
        if style == "analyst" and level != "short":
            base += f" قيمة الفرصة {xg:.2f} هدف متوقع."
        return base
    base = f"Chance missed! {s} shoots wide in minute {m}."
    if style == "analyst" and level != "short":
        base += f" The chance was worth {xg:.2f} expected goals."
    return base


def press(f: dict, lang: str, style: str, level: str) -> str:
    club, cnt = _club(f, "club", lang), f["count"]
    if lang == "ar":
        base = f"{club} يضغط عالي على خصمه."
        if style == "analyst" and level != "short":
            base += f" {cnt} ضغطات في آخر ثلاث دقايق."
        return base
    base = f"{club} are pressing high."
    if style == "analyst" and level != "short":
        base += f" {cnt} pressure events in the last three minutes."
    return base


def chaos(f: dict, lang: str, style: str, level: str) -> str:
    club, n = _club(f, "club", lang), f["shots"]
    if lang == "ar":
        base = f"فترة محمومة، {club} يسدد {n} تسديدات في دقيقتين."
        if style == "analyst" and level != "short":
            base += f" مجموع الفرص {f['xg_total']:.2f} هدف متوقع."
        return base
    base = f"A frantic spell, {club} fire {n} shots in two minutes."
    if style == "analyst" and level != "short":
        base += f" Combined chance value is {f['xg_total']:.2f} expected goals."
    return base


def ambient(f: dict, lang: str, style: str, level: str) -> str:
    club = _club(f, "club", lang)
    zone_ar = {"defensive": "في منطقتهم", "middle": "في وسط الملعب", "attacking": "قريب من منطقة الجزاء"}
    zone_en = {"defensive": "in their own half", "middle": "in midfield", "attacking": "near the box"}
    if lang == "ar":
        return f"الكرة مع {club} {zone_ar[f['zone']]}."
    return f"{club} have the ball {zone_en[f['zone']]}."


def ledger(f: dict, lang: str, style: str, level: str) -> str:
    """Investor-mode line: how much a moment moved the headline number vs the evidence."""
    p = _n(f, "player", lang)
    d_obs, d_ev = f["delta_observed"], f["delta_evidence"]
    if f["gate"] == "insufficient":
        if lang == "ar":
            unit = "تسديدة" if f["role"] != "GK" else "تسديدة مواجهة"
            return (f"{p}: الأدلة غير كافية للحكم على عقده. لحد الحين {f['n']} {unit}، "
                    f"ونحتاج {f['need']} بعد.")
        unit = "shots" if f["role"] != "GK" else "shots faced"
        return (f"{p}: not enough evidence to judge the contract yet, {f['n']} {unit} so far, "
                f"{f['need']} more needed.")
    lo, hi = _pct(f["luck_lo"]), _pct(f["luck_hi"])
    if lang == "ar":
        if level == "short":
            return f"{p}: الرقم الظاهر {d_obs:+.2f}، والأدلة {d_ev:+.2f}."
        return (f"{p}: الرقم الظاهر تحرك {d_obs:+.2f}، والتقدير المبني على الأدلة تحرك {d_ev:+.2f} فقط. "
                f"الحظ يفسر ما بين {lo} و{hi} بالمية من الفرق بين نتايجه وتوقعاته.")
    if level == "short":
        return f"{p}: headline {d_obs:+.2f}, evidence {d_ev:+.2f}."
    return (f"{p}: the headline number moved {d_obs:+.2f}, the evidence-based estimate only {d_ev:+.2f}. "
            f"Luck explains between {lo} and {hi} percent of the gap between his results and expectation.")


def state(f: dict, lang: str, style: str, level: str) -> str:
    a, b = _club(f, "a", lang), _club(f, "b", lang)
    m = f["minute"]
    if lang == "ar":
        base = f"الدقيقة {m}: {a} {f['ga']}، {b} {f['gb']}."
        if style in ("analyst", "investor") and level != "short":
            base += f" التسديدات {f['sa']} مقابل {f['sb']}، والأهداف المتوقعة {f['xa']:.2f} مقابل {f['xb']:.2f}."
        return base
    base = f"Minute {m}: {a} {f['ga']}, {b} {f['gb']}."
    if style in ("analyst", "investor") and level != "short":
        base += f" Shots {f['sa']} to {f['sb']}, expected goals {f['xa']:.2f} to {f['xb']:.2f}."
    return base


RENDERERS = {
    "goal": goal, "save": save, "missed": missed, "press": press,
    "chaos": chaos, "ambient": ambient, "ledger": ledger, "state": state,
}


def render(kind: str, facts: dict[str, Any], lang: str, style: str, level: str = "full") -> str:
    return RENDERERS[kind](facts, lang, style, level)


def catch_up_prefix(lang: str) -> str:
    return "قبل شوي: " if lang == "ar" else "A moment ago: "
