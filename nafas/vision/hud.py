"""Broadcast HUD layers rendered with Chromium (real Arabic shaping and font fallback), 1920x1080 RGBA."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

FONTS = Path(__file__).parent / "fonts"
W, H = 1920, 1080

HTML = """<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><style>
@font-face{font-family:Plex;src:url('FONTS/PlexArabic-Latin-Medium.ttf');unicode-range:U+0000-024F,U+2000-206F,U+2190-21FF;}
@font-face{font-family:Plex;src:url('FONTS/PlexArabic-Medium.ttf');unicode-range:U+0600-06FF,U+0750-077F,U+FB50-FDFF,U+FE70-FEFF;}
@font-face{font-family:Kufi;src:url('FONTS/ReemKufi-Latin-SemiBold.ttf');unicode-range:U+0000-024F,U+2000-206F;}
@font-face{font-family:Kufi;src:url('FONTS/ReemKufi-SemiBold.ttf');unicode-range:U+0600-06FF,U+FB50-FDFF,U+FE70-FEFF;}
:root{--turf:#0A3326;--deep:#06241B;--chalk:#EAF2EC;--chalk2:#A9C4B3;--flare:#FF6B4A;--mint:#4FD6A0;--gold:#F5B63A;--glass:rgba(6,36,27,.86)}
*{box-sizing:border-box;margin:0}
html,body{width:1920px;height:1080px;background:transparent;overflow:hidden;font-family:Plex,sans-serif;color:var(--chalk);font-weight:500}
.abs{position:absolute}
.glass{background:var(--glass);border:1.5px solid rgba(234,242,236,.16);border-radius:14px;box-shadow:0 8px 30px rgba(0,0,0,.35)}
#bug{left:48px;top:40px;height:68px;display:flex;align-items:center;direction:ltr;gap:0;padding:0 8px 0 0;overflow:hidden}
#bug .chip{width:12px;height:100%}
#bug .nm{font-family:Kufi,Plex;font-size:27px;padding:0 16px;white-space:nowrap}
#bug .sc{font-family:Kufi;font-size:42px;min-width:54px;text-align:center}
#bug .clk{font-family:Kufi;font-size:25px;background:var(--gold);color:var(--deep);border-radius:999px;padding:3px 16px;margin:0 8px}
#brand{right:56px;top:36px;display:flex;align-items:center;gap:14px;direction:rtl}
#brand .nm{font-family:Kufi;font-size:60px;line-height:1}
#brand .tag{font-size:22px;color:var(--chalk2);background:rgba(6,36,27,.8);border-radius:999px;padding:4px 16px;border:1px solid rgba(79,214,160,.5)}
#brand svg{width:64px;height:38px}
#metrics{left:48px;top:140px;width:352px;padding:16px 20px 14px}
#metrics h3{font-family:Kufi;font-size:25px;color:var(--mint);font-weight:600;margin-bottom:8px}
#metrics .r{display:flex;justify-content:space-between;align-items:baseline;padding:7px 0;border-top:1px solid rgba(234,242,236,.12);font-size:22px;color:var(--chalk2)}
#metrics .r b{font-family:Kufi;font-size:30px;color:var(--chalk);font-weight:600;direction:ltr}
#metrics .r b small{font-size:17px;color:var(--chalk2);margin-inline-start:5px}
#shotcard{left:50%;top:132px;transform:translateX(-50%);padding:14px 28px;min-width:720px;display:none;direction:rtl;text-align:center}
#shotcard .t{font-family:Kufi;font-size:36px}
#shotcard .o{display:inline-block;border-radius:8px;padding:2px 16px;margin-inline-start:14px;color:var(--deep);font-size:30px}
#shotcard .d{font-size:24px;color:var(--chalk2);margin-top:4px;direction:rtl}
#ledger{right:56px;top:140px;width:470px;padding:16px 22px 12px;display:none}
#ledger h3{font-family:Kufi;font-size:25px;color:var(--gold);font-weight:600;margin-bottom:6px}
#ledger .p{padding:9px 0;border-top:1px solid rgba(234,242,236,.12)}
#ledger .p .n{font-size:25px;display:flex;justify-content:space-between}
#ledger .p .n i{font-style:normal;font-family:Kufi;color:var(--chalk)}
#ledger .p .s{font-size:19px;color:var(--chalk2);margin-top:2px}
#ledger svg{display:block;margin-top:5px;direction:ltr}
#radar{right:56px;bottom:196px;width:352px;height:262px;padding:12px 16px}
#radar h3{font-family:Kufi;font-size:23px;color:var(--mint);font-weight:600}
#cap{left:48px;right:440px;bottom:196px;padding:18px 30px 16px;display:none;border-inline-start:9px solid var(--mint);direction:rtl}
#cap .who{font-size:21px;color:var(--chalk2);display:flex;gap:12px;align-items:center}
#cap .who span{background:rgba(79,214,160,.16);border-radius:999px;padding:1px 14px;color:var(--mint)}
#cap .ar{font-size:50px;line-height:1.5;margin-top:2px}
#cap .en{font-size:27px;color:var(--chalk2);direction:ltr;text-align:left;margin-top:2px}
#cap.long .ar{font-size:36px;line-height:1.45}#cap.long .en{font-size:21px}
#cap.goal{border-inline-start-color:var(--gold)}#cap.goal .who span{background:rgba(245,182,58,.18);color:var(--gold)}
#cap.cut{border-inline-start-color:var(--flare)}
#lanes{left:48px;right:48px;bottom:34px;height:150px;padding:12px 24px 10px;direction:ltr}
#lanes .row{position:absolute;left:24px;right:24px;height:34px}
#lanes .lab{position:absolute;right:0;top:3px;font-size:22px;color:var(--chalk2);direction:rtl;width:150px;text-align:right}
#lanes .track{position:absolute;left:0;right:170px;top:16px;height:2px;background:rgba(234,242,236,.2)}
#lanes .bar{position:absolute;top:6px;height:22px;border-radius:3px;background:var(--chalk)}
#lanes .bar.led{background:var(--mint)}#lanes .bar.cut{opacity:.55}
#lanes .cutm{position:absolute;top:-2px;width:0;height:0;border-left:7px solid transparent;border-right:7px solid transparent;border-top:11px solid var(--flare)}
#lanes .drop{position:absolute;top:10px;width:14px;height:14px;border:2px solid var(--chalk2);border-radius:50%}
#lanes .goalpin{position:absolute;top:-6px;width:3px;height:46px;background:var(--flare)}
#lanes h4{position:absolute;left:24px;top:-4px;font-size:20px;color:var(--chalk2);font-weight:400}
</style></head><body>
<div id="bug" class="abs glass"></div>
<div id="brand" class="abs"><svg viewBox="0 0 44 26"><path d="M1 13h9l3-11 5 22 5-16 3 5h18" fill="none" stroke="#4FD6A0" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/></svg><span class="nm">نَفَس</span><span class="tag" id="tag"></span></div>
<div id="metrics" class="abs glass"></div>
<div id="shotcard" class="abs glass"></div>
<div id="ledger" class="abs glass"></div>
<div id="radar" class="abs glass"><h3 id="radarT"></h3></div>
<div id="cap" class="abs glass"></div>
<div id="lanes" class="abs glass"></div>
</body></html>""".replace("FONTS", "")


class Hud:
    def __init__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch()
        self.page = self.browser.new_page(viewport={"width": W, "height": H})
        html = HTML.replace("url('", f"url('file://{FONTS}/")
        p = FONTS.parent / "_hud.html"
        p.write_text(html, encoding="utf-8")
        self.page.goto(f"file://{p}")
        self.page.wait_for_timeout(400)
        self.page.evaluate("document.fonts.ready")

    def close(self):
        self.browser.close()
        self._pw.stop()

    def render(self, js: str) -> np.ndarray:
        self.page.evaluate(js)
        png = self.page.screenshot(omit_background=True, type="png")
        return np.array(Image.open(io.BytesIO(png)).convert("RGBA"))


def state_js(s: dict) -> str:
    """One JS call that rewrites every HUD element from a state dict."""
    return "(" + _APPLY + ")(" + json.dumps(s, ensure_ascii=False) + ")"


_APPLY = r"""function(s){
 const $=id=>document.getElementById(id);
 const col={KBY:'#2E7BE0',RMR:'#E5392D'};
 $('bug').innerHTML=`<div class="chip" style="background:${col.KBY}"></div><div class="nm">${s.teams.KBY}</div><div class="sc">${s.score.KBY}</div><div class="clk">${s.clock}</div><div class="sc">${s.score.RMR}</div><div class="nm">${s.teams.RMR}</div><div class="chip" style="background:${col.RMR}"></div>`;
 $('tag').textContent=s.tag;
 const m=s.metrics;
 $('metrics').innerHTML=`<h3>${s.t.metrics}</h3>`+m.map(r=>`<div class="r"><span>${r[0]}</span><b>${r[1]}<small>${r[2]||''}</small></b></div>`).join('');
 $('radarT').textContent=s.t.radar;
 const sc=$('shotcard'); if(s.shot){sc.style.display='block'; sc.innerHTML=`<div class="t">${s.shot.title}<span class="o" style="background:${s.shot.color}">${s.shot.outcome}</span></div><div class="d">${s.shot.detail}</div>`;} else sc.style.display='none';
 const lg=$('ledger'); if(s.ledger){lg.style.display='block'; lg.innerHTML=`<h3>${s.t.ledger}</h3>`+s.ledger.map(p=>{
   const X=v=>8+Math.max(0,Math.min(1,v/2.2))*(334);
   const bar=p.gate==='insufficient'?`<line x1="${X(p.lo)}" x2="${X(p.hi)}" y1="11" y2="11" stroke="#A9C4B3" stroke-width="4" stroke-dasharray="4 4"/>`:`<rect x="${X(p.lo)}" y="5" width="${Math.max(4,X(p.hi)-X(p.lo))}" height="12" rx="6" fill="${p.color}" opacity=".85"/>`;
   return `<div class="p"><div class="n"><span>${p.name}</span><i>${p.obs}</i></div><div class="s">${p.sub}</div><svg width="350" height="22" viewBox="0 0 350 22"><rect x="8" y="10" width="334" height="2" fill="rgba(234,242,236,.25)"/><line x1="${X(1)}" x2="${X(1)}" y1="1" y2="21" stroke="#EAF2EC" stroke-width="2"/>${bar}<circle cx="${X(Math.max(0,p.obsv))}" cy="11" r="7" fill="#06241B" stroke="#EAF2EC" stroke-width="2.5"/></svg></div>`}).join('');} else lg.style.display='none';
 const cp=$('cap'); if(s.cap){cp.style.display='block'; cp.className='abs glass'+(s.cap.kind==='goal'?' goal':'')+(s.cap.cut?' cut':'')+(s.cap.ar.length>80?' long':''); cp.innerHTML=`<div class="who"><span>${s.cap.who}</span>${s.cap.min}</div><div class="ar">${s.cap.ar}</div>${s.cap.en?`<div class="en">${s.cap.en}</div>`:''}`;} else cp.style.display='none';
 const ln=$('lanes'); const W0=ln.clientWidth-48-170;
 let h=`<h4>${s.t.lanes}</h4>`;
 s.lanes.forEach((L,i)=>{h+=`<div class="row" style="top:${22+i*37}px"><div class="lab">${L.label}</div><div class="track"></div>`;
   L.bars.forEach(b=>{h+=`<div class="bar ${b.led?'led':''} ${b.cut?'cut':''}" style="left:${b.x0*W0}px;width:${Math.max(4,(b.x1-b.x0)*W0)}px"></div>`; if(b.cut) h+=`<div class="cutm" style="left:${b.x1*W0-7}px"></div>`;});
   L.drops.forEach(d=>{h+=`<div class="drop" style="left:${d*W0-7}px"></div>`;});
   s.goals.forEach(g=>{h+=`<div class="goalpin" style="left:${g*W0-1}px"></div>`;});
   h+='</div>';});
 ln.innerHTML=h;
}"""
