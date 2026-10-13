"""Title, problem, architecture and end cards for the submission video (1920x1080 PNGs, rendered in Chromium).

    PYTHONPATH=. python scripts/video_cards.py OUTDIR
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

FONTS = Path(__file__).resolve().parent.parent / "nafas/vision/fonts"
REPO = "github.com/tariqalshehri990-tech/gamesense-ai"

CSS = f"""
@font-face{{font-family:Plex;src:url('file://{FONTS}/PlexArabic-Latin-Medium.ttf')}}
@font-face{{font-family:Kufi;src:url('file://{FONTS}/ReemKufi-SemiBold.ttf');unicode-range:U+0600-06FF,U+FB50-FDFF,U+FE70-FEFF}}
:root{{--turf:#0A3326;--deep:#06241B;--chalk:#EAF2EC;--chalk2:#A9C4B3;--flare:#FF6B4A;--mint:#4FD6A0;--gold:#F5B63A}}
*{{box-sizing:border-box;margin:0}}
html,body{{width:1920px;height:1080px;overflow:hidden;background:var(--deep);color:var(--chalk);font-family:Plex,sans-serif}}
body{{background:repeating-linear-gradient(90deg,#06241B 0 120px,#07291F 120px 240px)}}
.pad{{position:absolute;inset:0;padding:110px 150px}}
h1{{font-family:Kufi,Plex;font-weight:600}}
.mint{{color:var(--mint)}}.gold{{color:var(--gold)}}.flare{{color:var(--flare)}}.dim{{color:var(--chalk2)}}
.box{{position:absolute;border:2px solid rgba(234,242,236,.35);background:rgba(10,51,38,.92);border-radius:14px;padding:16px 22px;font-size:30px;line-height:1.25}}
.box small{{display:block;font-size:22px;color:var(--chalk2);margin-top:4px}}
.box.mintb{{border-color:var(--mint)}}.box.goldb{{border-color:var(--gold)}}.box.flareb{{border-color:var(--flare)}}
.hide{{opacity:0}}
svg.arrows{{position:absolute;inset:0}}
"""


def page(body: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"


def title() -> str:
    return page("""<div class='pad' style='display:flex;flex-direction:column;justify-content:center;align-items:center;gap:20px'>
<svg width='220' height='130' viewBox='0 0 44 26'><path d='M1 13h9l3-11 5 22 5-16 3 5h18' fill='none' stroke='#4FD6A0' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/></svg>
<h1 style='font-size:230px;line-height:1'>طارق</h1>
<h1 style="font-size:92px;letter-spacing:.04em">Tariq</h1>
<p style='font-size:44px;color:var(--chalk2);margin-top:14px'>The star that cuts through the noise</p></div>""")


def problem(n: int) -> str:
    rows = [("A viewer can only follow so much", "Commentators talk about everything, all the time.", "mint"),
            ("A blind fan cannot rewind", "Miss the goal for a side comment and it is gone.", "gold"),
            ("A club pays for luck", "A hot streak looks like skill until you ask how many shots.", "flare")]
    out = ""
    for i, (a, b, c) in enumerate(rows):
        op = 1 if i < n else 0
        out += f"<div style='opacity:{op};margin-bottom:64px'><div class='{c}' style='font-family:Kufi,Plex;font-size:78px'>{a}</div><div style='font-size:40px;color:var(--chalk2);margin-top:6px'>{b}</div></div>"
    return page(f"<div class='pad' style='display:flex;flex-direction:column;justify-content:center'>{out}</div>")


def arch(stage: int) -> str:
    def b(x, y, w, h, t, s="", cls="", show=1):
        return f"<div class='box {cls}' style='left:{x}px;top:{y}px;width:{w}px;height:{h}px;opacity:{1 if show else 0}'>{t}<small>{s}</small></div>"
    s1, s2, s3 = stage >= 1, stage >= 2, stage >= 3
    body = "<div class='pad'><h1 style='font-size:62px'>One match, four audiences</h1></div>"
    body += b(90, 200, 330, 130, "Synthetic match", "football-realistic events", "mintb", s1)
    body += b(520, 200, 330, 130, "Rendered video", "broadcast-style camera", "", s1)
    body += b(950, 200, 330, 130, "Vision tracker", "players + ball, calibrated", "", s1)
    body += b(1380, 200, 400, 130, "Events + xG", "pass, shot, save, goal", "mintb", s1)
    body += b(90, 470, 330, 130, "Facts over MCP", "one source of truth", "goldb", s2)
    body += b(520, 440, 330, 190, "Conductor", "shared state, preempts, hands off", "goldb", s2)
    for i, (n, d) in enumerate([("Scout", "finds moments"), ("Explainer", "writes the line"), ("Editor", "cuts or compresses"), ("DialectDesk", "Arabic / English"), ("Auditor", "checks every number")]):
        body += b(950 + (i % 2) * 400, 410 + (i // 2) * 110, 370, 90, n, d, "", s2)
    body += b(90, 800, 400, 150, "Casual fan", "short, one thing at a time", "mintb", s3)
    body += b(530, 800, 400, 150, "Audio description", "dense, positional", "mintb", s3)
    body += b(970, 800, 400, 150, "Analyst", "evidence on every line", "mintb", s3)
    body += b(1410, 800, 400, 150, "Investor", "skill vs luck ledger", "goldb", s3)
    a = ""
    if s1:
        a += "<path d='M420 265H520M850 265H950M1280 265H1380' stroke='#A9C4B3' stroke-width='3' fill='none'/>"
    if s2:
        a += "<path d='M1580 330V380H255V470M420 535H520M850 535H950' stroke='#F5B63A' stroke-width='3' fill='none'/>"
    if s3:
        a += "<path d='M685 630V770H290V800M685 770H730V800M685 770H1170V800M685 770H1610V800' stroke='#4FD6A0' stroke-width='3' fill='none'/>"
    body += f"<svg class='arrows' width='1920' height='1080'>{a}</svg>"
    return page(body)


def end() -> str:
    items = [("Microsoft Agent Framework + Foundry", "optional phrasing backend", "gold"),
             ("Model Context Protocol", "facts served to any agent", "mint"),
             ("Azure Container Apps", "Dockerfile ships the app", "mint"),
             ("Azure AI Speech", "Arabic and English voice", "mint")]
    rows = "".join(f"<div style='display:flex;align-items:baseline;gap:26px;margin-bottom:26px'><span class='{c}' style='font-family:Kufi,Plex;font-size:62px'>{a}</span><span style='font-size:34px;color:var(--chalk2)'>{b}</span></div>" for a, b, c in items)
    return page(f"""<div class='pad'><h1 style='font-size:84px;margin-bottom:46px'>Built with</h1>{rows}
<div style='position:absolute;left:150px;bottom:130px'><div style='font-size:50px'>{REPO}</div>
<div class='gold' style='font-size:36px;margin-top:12px'>All data is synthetic. No real match footage.</div></div></div>""")


def badge() -> str:
    return ("<!doctype html><html><head><meta charset='utf-8'><style>" + CSS.replace("html,body{width:1920px;height:1080px;overflow:hidden;background:var(--deep)", "html,body{width:520px;height:64px;overflow:hidden;background:transparent")
            + "body{background:transparent!important}</style></head><body><div style='display:inline-block;margin:6px;background:rgba(6,36,27,.92);border:2px solid #F5B63A;color:#F5B63A;border-radius:999px;padding:8px 22px;font-size:26px'>Synthetic data · no real footage</div></body></html>")


def main(out: str) -> None:
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    cards = {"title": title(), "problem1": problem(1), "problem2": problem(2), "problem3": problem(3),
             "arch1": arch(1), "arch2": arch(2), "arch3": arch(3), "end": end()}
    with sync_playwright() as p:
        br = p.chromium.launch()
        pg = br.new_page(viewport={"width": 1920, "height": 1080})
        for k, h in cards.items():
            f = o / f"_{k}.html"
            f.write_text(h, encoding="utf-8")
            pg.goto(f"file://{f}")
            pg.wait_for_timeout(500)
            pg.screenshot(path=str(o / f"{k}.png"))
        pg2 = br.new_page(viewport={"width": 520, "height": 64})
        (o / "_badge.html").write_text(badge(), encoding="utf-8")
        pg2.goto(f"file://{o / '_badge.html'}")
        pg2.wait_for_timeout(500)
        pg2.screenshot(path=str(o / "badge.png"), omit_background=True)
        br.close()


if __name__ == "__main__":
    main(sys.argv[1])
