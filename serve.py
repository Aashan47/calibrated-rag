"""A zero-dependency web demo: ask the agent a question in the browser and watch it answer
(with a citation and a confidence bar) or abstain.

    GEMINI_API_KEY=... python serve.py           # then open http://localhost:8000

Pure Python standard library — no web framework. The /ask endpoint runs the same agent the
evaluation uses, with the calibrated abstention threshold from results/results.json.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import time

from calibrated_rag import agent as agent_mod
from calibrated_rag import corpus, trace

print("Loading corpus + agent...")
_CONTEXTS, _SOURCE, _CUSTOM = corpus.load_contexts()
_AGENT = agent_mod.Agent(_CONTEXTS)
_TAU = corpus.abstention_threshold(_CUSTOM)
print(f"Ready: {len(_CONTEXTS)} passages from {_SOURCE}; abstention threshold = {_TAU:.2f}")


def _format_steps(steps: list[dict]) -> list[str]:
    """Turn the agent's raw action log into short human-readable lines for the UI."""
    out = []
    for s in steps or []:
        a = s.get("action")
        if a == "search":
            out.append(f"🔎 searched “{s.get('query','')}” "
                       f"→ {s.get('new', 0)} new passage(s)")
        elif a == "decide":
            if s.get("next") == "search":
                out.append(f"🤔 passages insufficient → reformulating: “{s.get('query','')}”")
            else:
                out.append("✅ judged passages sufficient → answering")
        elif a == "abstain":
            out.append("🛑 judged the answer isn't in this corpus → abstaining")
    return out


def _answer(question: str) -> dict:
    t0 = time.time()
    pred = _AGENT.predict(question)
    d = agent_mod.decide(pred, _TAU)
    steps = pred.get("steps", [])
    trace.log({"q": question[:200], "answered": d["answered"],
               "confidence": round(d["confidence"], 2),
               "steps": len([s for s in steps if s.get("action") == "search"]),
               "latency_ms": round((time.time() - t0) * 1000)})
    reason = ""
    if not d["answered"]:
        reason = ("the agent judged it unanswerable from these documents"
                  if not pred.get("answerable")
                  else f"confidence {pred.get('confidence', 0):.2f} is below the calibrated "
                       f"threshold of {_TAU:.2f}")
    cite = d.get("citation")
    return {"answered": d["answered"], "answer": d["answer"],
            "confidence": round(d["confidence"], 2), "threshold": round(_TAU, 2),
            "reason": reason, "citation": (cite or {}).get("text", "") if cite else "",
            "trace": _format_steps(steps)}


PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>calibrated-rag</title>
<style>
:root{--bg:#0f1720;--card:#17212b;--ink:#e7edf3;--mut:#93a1b1;--line:#243140;--accent:#4da3ff;
--ok:#43d6a0;--warn:#f0b64b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:720px;margin:0 auto;padding:40px 20px 80px}
h1{font-size:24px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 24px}
.sub b{color:var(--ink)}
.box{display:flex;gap:10px}
input{flex:1;padding:12px 14px;border-radius:10px;border:1px solid var(--line);
background:var(--card);color:var(--ink);font-size:15px}
input:focus{outline:none;border-color:var(--accent)}
button{padding:12px 18px;border:0;border-radius:10px;background:var(--accent);color:#06121f;
font-weight:600;cursor:pointer}button:disabled{opacity:.5;cursor:default}
.chips{margin:14px 0 0;display:flex;flex-wrap:wrap;gap:8px}
.chip{font-size:13px;color:var(--mut);border:1px solid var(--line);border-radius:20px;
padding:5px 12px;cursor:pointer;background:transparent}.chip:hover{color:var(--ink);border-color:var(--accent)}
.card{margin-top:22px;border:1px solid var(--line);border-radius:14px;padding:18px 20px;background:var(--card);display:none}
.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut)}
.ans{font-size:20px;font-weight:600;margin:6px 0 2px}
.bar{height:8px;border-radius:6px;background:var(--line);margin:14px 0 6px;overflow:hidden}
.fill{height:100%;background:var(--accent)}
.meta{font-size:13px;color:var(--mut)}
.cite{margin-top:14px;padding:12px 14px;border-left:3px solid var(--accent);background:#0d151e;
border-radius:6px;font-size:13.5px;color:#c9d6e2;max-height:150px;overflow:auto}
.trace{margin-top:14px;padding:12px 14px;background:#0d151e;border-radius:8px;font-size:13px;
color:var(--mut);border:1px solid var(--line)}
.trace .lbl{margin-bottom:6px}.trace ol{margin:0;padding-left:18px}.trace li{margin:3px 0}
.pill{display:inline-block;font-size:12px;font-weight:600;padding:3px 10px;border-radius:20px}
.pill.ok{background:rgba(67,214,160,.14);color:var(--ok)}
.pill.ab{background:rgba(240,182,75,.14);color:var(--warn)}
.foot{margin-top:28px;color:var(--mut);font-size:12.5px}
</style></head><body><div class=wrap>
<h1>calibrated-rag</h1>
<p class=sub>A retrieval-QA agent that <b>knows when to abstain</b>. Ask something answerable from
the corpus and it answers with a citation and a calibrated confidence; ask something it can't
support and it abstains instead of guessing.</p>
<div class=box><input id=q placeholder="Ask a question..." autofocus>
<button id=go onclick=ask()>Ask</button></div>
<div class=chips id=chips></div>
<div class=card id=card></div>
<p class=foot id=foot></p>
</div><script>
const EX=["Who was yersinia pestis named for?","Who is seen as the ultimate climate change authority?","What is the capital of Mars?","Who won the 2050 World Cup?"];
const chips=document.getElementById('chips');
EX.forEach(e=>{const c=document.createElement('span');c.className='chip';c.textContent=e;
c.onclick=()=>{document.getElementById('q').value=e;ask()};chips.appendChild(c)});
async function ask(){
 const q=document.getElementById('q').value.trim(); if(!q)return;
 const go=document.getElementById('go'),card=document.getElementById('card');
 go.disabled=true;card.style.display='block';card.innerHTML='<span class=meta>agent working (searching, judging, then self-consistency sampling)...</span>';
 try{
  const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  const d=await r.json(); render(d);
 }catch(e){card.innerHTML='<span class=meta>error: '+e+'</span>'}
 go.disabled=false;
}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function traceHTML(t){
 if(!t||!t.length)return '';
 return `<div class=trace><div class=lbl>agent trace</div><ol>`+
   t.map(s=>`<li>${esc(s)}</li>`).join('')+`</ol></div>`;
}
function render(d){
 const card=document.getElementById('card'); const pct=Math.round(d.confidence*100);
 if(d.answered){
  card.innerHTML=`<span class="pill ok">ANSWERED</span>
   <div class=ans>${esc(d.answer)}</div>
   <div class=bar><div class=fill style="width:${pct}%"></div></div>
   <div class=meta>confidence ${d.confidence.toFixed(2)} &nbsp;·&nbsp; calibrated threshold ${d.threshold.toFixed(2)}</div>
   ${d.citation?`<div class=lbl style="margin-top:14px">cited passage</div><div class=cite>${esc(d.citation)}</div>`:''}
   ${traceHTML(d.trace)}`;
 }else{
  card.innerHTML=`<span class="pill ab">ABSTAINED</span>
   <div class=ans style="font-size:17px">I can't answer this reliably from the documents.</div>
   <div class=meta>reason: ${esc(d.reason)}</div>
   ${traceHTML(d.trace)}`;
 }
}
fetch('/meta').then(r=>r.json()).then(m=>{document.getElementById('foot').textContent=
 `${m.n} passages · corpus: ${m.source} · model: ${m.model} · abstains below confidence ${m.threshold}`});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/meta":
            self._send(200, json.dumps({"n": len(_CONTEXTS), "source": _SOURCE,
                                        "threshold": round(_TAU, 2),
                                        "model": os.environ.get("CRAG_MODEL", "gemini-2.5-flash")}))
        else:
            self._send(404, "{}")

    def do_POST(self):  # noqa: N802
        if self.path != "/ask":
            self._send(404, "{}")
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            q = json.loads(self.rfile.read(n) or b"{}").get("question", "").strip()
            self._send(200, json.dumps(_answer(q)) if q else "{}")
        except Exception as exc:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(exc)}))

    def log_message(self, *a):  # quiet
        pass


def main():
    port = int(os.environ.get("PORT", "8000"))
    print(f"Serving on http://localhost:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
