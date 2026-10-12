"""Build demo/index.html: the UI plus seed-35 stream data and what-if results embedded (no server needed)."""
import json
from pathlib import Path
from fastapi.testclient import TestClient
from nafas.attention import PROFILES
from nafas.server import _facts, app

ROOT = Path(__file__).resolve().parent.parent
c = TestClient(app)
streams = {}
for p in PROFILES:
    for lang in ("ar", "en"):
        r = c.get(f"/api/stream?seed=35&profile={p}&lang={lang}&speed=1000000")
        streams[f"{p}|{lang}"] = [json.loads(l[6:]) for l in r.text.split("\n\n") if l.startswith("data: ")]
ids = {i["shot_id"] for s in streams.values() for i in s if i.get("shot_id")}
wi = {}
for sid in ids:
    for g in (True, False):
        res = _facts(35).whatif(sid, g)
        if "error" not in res:
            wi[f"{sid}|{g}"] = res
shim = """<script>
window.__DEMO=%s;
(function(){
 const D=window.__DEMO;
 window.EventSource=function(url){
  const q=new URL(url,location.href).searchParams; const key=q.get('profile')+'|'+(q.get('lang')||'ar');
  const items=D.streams[key]||[]; const speed=Number(q.get('speed'))||60; const self=this; let dead=false,last=0,i=0;
  this.close=()=>{dead=true};
  function next(){ if(dead||i>=items.length) return; const it=items[i++]; const dt=Math.min(Math.max((it.t-last)/speed,0),4); last=Math.max(last,it.t||0);
    setTimeout(()=>{ if(dead) return; self.onmessage&&self.onmessage({data:JSON.stringify(it)}); next(); }, dt*1000); }
  setTimeout(next,50);
 };
 const of=window.fetch;
 window.fetch=function(u,o){ if(String(u).includes('/api/whatif')){ const b=JSON.parse(o.body); const r=D.whatif[b.shot_id+'|'+(b.goal?'True':'False')]||{error:'n/a'};
   return Promise.resolve({ok:true,json:()=>Promise.resolve(r)}); } return of.apply(this,arguments); };
})();
</script>""" % json.dumps({"streams": streams, "whatif": wi}, ensure_ascii=False)
html = (ROOT / "nafas/static/index.html").read_text()
audio = ROOT / "demo/audio.json"
if audio.exists():  # pre-rendered clips (scripts/build_audio.py) so the demo speaks on any device
    shim += "\n<script>window.__AUDIO=" + audio.read_text() + ";</script>"
(ROOT / "demo").mkdir(exist_ok=True)
(ROOT / "demo/index.html").write_text(html.replace("<script>", shim + "\n<script>", 1))
print("demo built:", len(streams), "streams,", len(wi), "what-ifs")
