"""Build the blinded annotation page for the error pilot (or any later human evaluation).

    python scripts/07_make_annotation.py --name pilot --n 100 --systems draft
    python scripts/07_make_annotation.py --name pilot2 --n 100 --systems draft ours b0plus

Writes annotation/<name>/annotate.html (self-contained, works offline) and
annotation/<name>/key.json (which system produced which caption -- do NOT open
it before all labels are in). Each unique (image, caption) pair appears once;
items are shuffled so that versions of one image are not adjacent.

Look L1 (docs/LOOK_L1_PROTOKOL.md, step 9): every claim marked "incorrect" gets a second,
optional field -- misidentified (the object is there, named wrong) vs invented (no such
object) -- and misidentified claims get a drag-to-draw box around the true referent. This
turns the pilot into Look L1's gold-standard test set; scripts/10_score_labels.py reads both
new columns. Add these fields before launching the already-planned Check human pilot
(docs/CALISMA_GUNLUGU.md, "Sıradaki adımlar" item 0), not after, so one annotation round
covers both -- see docs/LOOK_L1_PROTOKOL.md Section 8.

B human pilot (--pilot lcr, docs/PILOT_PROTOKOL.md): claims are sampled from the current S4 and S2 records
(A_B_lcr4.json, A_B_lcr2.json) by lcr/pilot.pilot_sample -- a random group for natural error rates, every S4
replace/remove, S2-contradicted S4-keeps (clothing first), keep_elsewhere and a keep base rate. One card per
image; annotators label every noun as usual and never see which nouns were sampled or why. Only AFTER a sampled
noun is labelled: if S4 would replace it, "is <substitute> the right name for this object?"; if the noun has a
referent (correct, or misidentified), a box around it (for the detector / Look / human box comparison).

    python scripts/07_make_annotation.py --pilot lcr --name pilot_b
"""
import argparse
import base64
import io
import json

import nltk
import numpy as np

import _common  # noqa: F401
from lcr import cache, claims, config, data, pilot
from lcr import region_evidence as re_

NOT_OBJECTS = set("front top air background side middle back end edge way lot bunch pair kind day night "
                  "time picture photo view trick game part piece distance area set".split())

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="pilot")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--split", default="validation")
ap.add_argument("--decoder", default="A")
ap.add_argument("--systems", nargs="+", default=["draft"])
ap.add_argument("--seed", type=int, default=config.SEED)
ap.add_argument("--pilot", choices=["lcr"], default=None, help="B human pilot sampled from S4/S2 decisions")
ap.add_argument("--n-random", type=int, default=60, help="pilot: uniform random claims (natural error rates)")
ap.add_argument("--n-s2-keep", type=int, default=80, help="pilot: S2 contradicted but S4 keep (clothing first)")
ap.add_argument("--n-elsewhere", type=int, default=60, help="pilot: S4 keep_elsewhere")
ap.add_argument("--n-keep", type=int, default=80, help="pilot: S4 keep (base rate of errors S4 lets through)")
ap.add_argument("--allow-unrewritten-s4", action="store_true",
                help="pilot: use A_B_lcr4.json although scripts/48 has not rewritten it (old evidence boxes)")
a = ap.parse_args()
assert a.split != "test" or a.name.startswith("final"), "test images only for the final evaluation"
rng = np.random.default_rng(a.seed)
focus, population = [], None

if a.pilot:
    P = config.CACHE / "pipeline"
    rewrite = config.RESULTS / "lcr4_witness_rewrite.json"
    if not (rewrite.exists() and cache.load_json(rewrite).get("written")) and not a.allow_unrewritten_s4:
        ap.error("A_B_lcr4.json has not been rewritten by scripts/48 (S4 evidence-box fix, docs/CHECK_S4.md); "
                 "run it first or pass --allow-unrewritten-s4")
    s4_rows, s2_rows = cache.load_json(P / "A_B_lcr4.json"), cache.load_json(P / "A_B_lcr2.json")
    assert all(r["id"].startswith("train_") for r in s4_rows), "the pilot is on B only"
    s4_bundle = cache.load_json(re_.BUNDLE_PATH)["id"]
    assert all(r["bundle_id"] == s4_bundle for r in s4_rows), "A_B_lcr4.json is not from the current S4 bundle"
    focus, population = pilot.pilot_sample(pilot.pilot_pool(s4_rows, s2_rows), rng, n_random=a.n_random,
                                           n_s2_keep=a.n_s2_keep, n_elsewhere=a.n_elsewhere, n_keep=a.n_keep)
    rows = {r["id"]: r for r in s4_rows}
    sample = sorted({c["id"] for c in focus})
    a.systems, a.split = ["draft"], "B"
else:
    rows = {r["id"]: r for r in cache.load_json(config.CACHE / "pipeline" / f"{a.decoder}_{a.split}.json")}
    sample = sorted(rng.choice(sorted(rows), size=a.n, replace=False).tolist())

captions: dict[str, dict[str, str]] = {"draft": {i: rows[i]["draft"] for i in sample}}
for s in a.systems:
    if s != "draft":
        outs = cache.load_json(config.CACHE / "pipeline" / f"retell_{s}_{a.split}_outputs.json")
        captions[s] = {i: outs["real"][i] for i in sample}


def nouns(caption: str):
    ws = claims.words(caption)
    vocab = {c.index: c.cls for c in claims.extract_from_words(ws, dedupe=False)}
    out = []
    for i, (w, t) in enumerate(nltk.pos_tag(ws)):
        if i in vocab:
            out.append(dict(word=w, cls=vocab[i], in_vocab=True))
        elif t in ("NN", "NNS") and w not in NOT_OBJECTS and claims.word_class(w) is None:
            out.append(dict(word=w, cls=w, in_vocab=False))
    return out


items, key = [], {}
for iid in sample:
    seen: dict[str, str] = {}
    for s in a.systems:
        cap = captions[s][iid]
        if cap not in seen:
            seen[cap] = f"{len(items):04d}"
            items.append(dict(item="", image=iid, caption=cap, nouns=nouns(cap)))
        key.setdefault(iid, {})[s] = cap
order = rng.permutation(len(items))
# keep versions of the same image apart: re-draw the shuffle until no two neighbours share an image
for _ in range(200):
    if all(items[order[i]]["image"] != items[order[i + 1]]["image"] for i in range(len(order) - 1)):
        break
    order = rng.permutation(len(items))
items = [items[j] for j in order]
for n, it in enumerate(items):
    it["item"] = f"{a.name}-{n:04d}"

# pilot: attach each sampled claim to its noun (first in-vocabulary mention of the class, as claim extraction does)
item_of = {it["image"]: it for it in items}
focus_out, unmatched = [], 0
for c in focus:
    it = item_of[c["id"]]
    j = next((j for j, n in enumerate(it["nouns"]) if n["in_vocab"] and n["cls"] == c["cls"]), None)
    if j is None:
        unmatched += 1
        continue
    it["nouns"][j]["ask_box"] = True
    if c["s4_action"] == "replace" and c["substitute"]:
        it["nouns"][j]["ask_substitute"] = c["substitute"]
    focus_out.append(dict(c, item=it["item"], noun_index=j))


def thumb(iid: str) -> str:
    im = data.load_image(iid); im.thumbnail((520, 520))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


payload = [dict(item=it["item"], caption=it["caption"], nouns=it["nouns"], img=thumb(it["image"])) for it in items]
html = """<!doctype html><html lang="tr"><meta charset="utf-8"><title>LCR etiketleme: __NAME__</title>
<style>
body{font-family:system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;color:#222}
.card{border:1px solid #ccc;border-radius:8px;padding:12px;margin:16px 0;display:flex;gap:16px;flex-wrap:wrap}
.card img{max-width:420px;width:100%;border-radius:4px}.cap{font-size:18px;margin-bottom:8px}
.noun{margin:6px 0;padding:4px 6px;border-radius:4px}.noun.todo{background:#fff4d6}
.noun b{display:inline-block;min-width:110px}label{margin-right:10px;white-space:nowrap}
.errwrap{margin:6px 0 2px 0;padding:6px 8px;border-left:3px solid #e8a33d;background:#fff9ef}
.boxwrap{margin-top:6px}.boxhint{color:#666;font-size:12px;margin:2px 0}
.bar{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ccc;z-index:2}
button{font-size:15px;padding:6px 12px}small{color:#666}
</style>
<h2>Look-Check-Retell: nesne etiketleme (__NAME__)</h2>
<p><b>Kurallar.</b> Her alt&#305;n&#305; &ccedil;izili isim i&ccedil;in g&ouml;rsele bak&#305;n:
<b>do&#287;ru</b> = bu nesne g&ouml;rselde var ve do&#287;ru adland&#305;r&#305;lm&#305;&#351;;
<b>yanl&#305;&#351;</b> = b&ouml;yle bir nesne yok ya da ba&#351;ka bir nesne bu adla an&#305;lm&#305;&#351;;
<b>emin de&#287;il</b> = g&ouml;rselden karar verilemiyor;
<b>nesne de&#287;il</b> = kelime somut bir nesne de&#287;il (etiketleyici hatas&#305;).
&#304;nsan kelimeleri (man, woman, boy, girl...) yaln&#305;zca <i>bir insan var m&#305;</i> diye de&#287;erlendirilir;
cinsiyet ve ya&#351; de&#287;erlendirilmez. Eylemler, renkler ve say&#305;lar de&#287;erlendirilmez.
<b>Yanl&#305;&#351;</b> dedi&#287;iniz her claim i&ccedil;in ayr&#305;ca: <b>yanl&#305;&#351; tan&#305;mlanm&#305;&#351;</b>
(nesne g&ouml;rselde var, ba&#351;ka ad verilmi&#351;) mi yoksa <b>uydurulmu&#351;</b> (b&ouml;yle bir nesne yok) mu
oldu&#287;unu i&#351;aretleyin; yanl&#305;&#351; tan&#305;mlanm&#305;&#351;sa g&ouml;r&uuml;ntü &uuml;zerine ger&ccedil;ek
nesnenin &ccedil;evresine bir kutu &ccedil;izin (fareyle s&uuml;r&uuml;kleyin).
Baz&#305; isimlerde etiketten sonra ek sorular &ccedil;&#305;kar: &ouml;nerilen ba&#351;ka bir ad&#305;n bu nesne i&ccedil;in
do&#287;ru olup olmad&#305;&#287;&#305; ya da kelimenin g&ouml;sterdi&#287;i nesnenin &ccedil;evresine bir kutu. Bu sorular
etiketinizi de&#287;i&#351;tirmek i&ccedil;in de&#287;ildir; &ouml;nce kendi karar&#305;n&#305;z&#305; verin.</p>
<div class="bar">Etiketleyen: <input id="who" placeholder="ad"> &nbsp; <span id="prog"></span> &nbsp;
<button onclick="exportCsv()">CSV indir</button> <small>Yan&#305;tlar taray&#305;c&#305;da otomatik saklan&#305;r.</small></div>
<div id="list"></div>
<script>
const NAME="__NAME__", DATA=__DATA__;
const OPTS=[["correct","doğru"],["incorrect","yanlış"],["uncertain","emin değil"],["na","nesne değil"]];
const ERR_OPTS=[["misidentified","yanlış tanımlanmış"],["invented","uydurulmuş"]];
let raw=JSON.parse(localStorage.getItem("lcr-"+NAME)||"{}");
let ans={}; for(const k in raw) ans[k]=(typeof raw[k]==="string")?{label:raw[k]}:raw[k];   // migrate old string format
const who=document.getElementById("who"); who.value=localStorage.getItem("lcr-who")||""; who.oninput=()=>localStorage.setItem("lcr-who",who.value);
function findItem(k){return DATA.find(d=>d.item===k.split("|")[0])}
function findNoun(k){return findItem(k).nouns[+k.split("|")[1]]}
function save(){localStorage.setItem("lcr-"+NAME,JSON.stringify(ans))}
const SUB_OPTS=[["yes","evet"],["no","hayır"],["uncertain","emin değil"]];
const HINT_MIS="Ger&ccedil;ek nesnenin &ccedil;evresine bir kutu &ccedil;izin:", HINT_REF="Bu kelimenin g&ouml;sterdi&#287;i nesnenin &ccedil;evresine bir kutu &ccedil;izin:";
function wantsBox(n,rec){return rec.error_type==="misidentified"||(n.ask_box&&rec.label==="correct")}
function render(){const L=document.getElementById("list");L.innerHTML="";DATA.forEach(d=>{const c=document.createElement("div");c.className="card";
 let h='<img src="data:image/jpeg;base64,'+d.img+'"><div><div class="cap">'+d.caption+'</div><small>'+d.item+'</small>';
 d.nouns.forEach((n,j)=>{const k=d.item+"|"+j;const rec=ans[k]||{};
  h+='<div class="noun '+(rec.label?"":"todo")+'" id="n-'+k+'"><b><u>'+n.word+'</u></b>';
  OPTS.forEach(o=>{h+='<label><input type="radio" name="'+k+'" value="'+o[0]+'" '+(rec.label===o[0]?"checked":"")+' onchange="setLabel(\\''+k+'\\',this.value)">'+o[1]+'</label>'});
  h+='<div class="errwrap" id="errwrap-'+k+'" style="display:'+(rec.label==="incorrect"?"block":"none")+'">';
  ERR_OPTS.forEach(o=>{h+='<label><input type="radio" name="'+k+'-err" value="'+o[0]+'" '+(rec.error_type===o[0]?"checked":"")+' onchange="setErr(\\''+k+'\\',this.value)">'+o[1]+'</label>'});
  h+='</div>';
  if(n.ask_substitute){h+='<div class="errwrap" id="subwrap-'+k+'" style="display:'+(rec.label&&rec.label!=="na"?"block":"none")+'">&Ouml;nerilen ad: <b>'+n.ask_substitute+'</b>. Bu nesne i&ccedil;in do&#287;ru mu? ';
   SUB_OPTS.forEach(o=>{h+='<label><input type="radio" name="'+k+'-sub" value="'+o[0]+'" '+(rec.substitute_ok===o[0]?"checked":"")+' onchange="setSub(\\''+k+'\\',this.value)">'+o[1]+'</label>'});
   h+='</div>'}
  h+='<div class="boxwrap" id="boxwrap-'+k+'"></div></div>'});
 if(!d.nouns.length)h+='<small>(nesne ismi yok)</small>';c.innerHTML=h+'</div>';L.appendChild(c)});
 DATA.forEach(d=>d.nouns.forEach((n,j)=>{const k=d.item+"|"+j;const rec=ans[k]||{};if(wantsBox(n,rec))mountBox(k,rec.error_type==="misidentified"?HINT_MIS:HINT_REF)}));
 prog()}
function setLabel(k,v){ans[k]=ans[k]||{};const rec=ans[k],n=findNoun(k);rec.label=v;
 if(v!=="incorrect")rec.error_type=null;
 rec.box=null;document.getElementById("boxwrap-"+k).innerHTML="";
 if(wantsBox(n,rec))mountBox(k,HINT_REF);
 const sw=document.getElementById("subwrap-"+k);if(sw)sw.style.display=(v!=="na")?"block":"none";
 save();document.getElementById("n-"+k).classList.remove("todo");
 document.getElementById("errwrap-"+k).style.display=(v==="incorrect")?"block":"none";prog()}
function setErr(k,v){ans[k]=ans[k]||{};ans[k].error_type=v;const bw=document.getElementById("boxwrap-"+k);
 if(v==="misidentified"){mountBox(k,HINT_MIS)}else{bw.innerHTML="";ans[k].box=null}
 save()}
function setSub(k,v){ans[k]=ans[k]||{};ans[k].substitute_ok=v;save()}
function mountBox(k,hint){const wrap=document.getElementById("boxwrap-"+k);const d=findItem(k);const rec=ans[k]=ans[k]||{};
 wrap.innerHTML='<div class="boxhint">'+hint+'</div>';
 const holder=document.createElement("div");holder.style.cssText="position:relative;display:inline-block;max-width:280px";
 const img=document.createElement("img");img.src="data:image/jpeg;base64,"+d.img;img.style.cssText="display:block;width:100%;border-radius:4px";
 const cv=document.createElement("canvas");cv.style.cssText="position:absolute;top:0;left:0;width:100%;height:100%;cursor:crosshair";
 holder.appendChild(img);holder.appendChild(cv);wrap.appendChild(holder);
 const clr=document.createElement("button");clr.textContent="kutuyu temizle";clr.style.cssText="margin-left:8px;font-size:12px;padding:2px 6px";
 clr.onclick=()=>{rec.box=null;draw();save()};wrap.appendChild(clr);
 function size(){cv.width=cv.clientWidth;cv.height=cv.clientHeight;draw()}
 function draw(){const ctx=cv.getContext("2d");ctx.clearRect(0,0,cv.width,cv.height);
  if(rec.box){const[x0,y0,x1,y1]=rec.box;ctx.strokeStyle="#e33";ctx.lineWidth=2;
   ctx.strokeRect(x0*cv.width,y0*cv.height,(x1-x0)*cv.width,(y1-y0)*cv.height)}}
 img.onload=size;if(img.complete)size();
 let drag=null;
 cv.onmousedown=e=>{drag=[e.offsetX/cv.width,e.offsetY/cv.height]};
 cv.onmousemove=e=>{if(!drag)return;const x=Math.max(0,Math.min(1,e.offsetX/cv.width)),y=Math.max(0,Math.min(1,e.offsetY/cv.height));
  rec.box=[Math.min(drag[0],x),Math.min(drag[1],y),Math.max(drag[0],x),Math.max(drag[1],y)];draw()};
 window.addEventListener("mouseup",()=>{if(drag){drag=null;save()}})}
function total(){return DATA.reduce((s,d)=>s+d.nouns.length,0)}
function prog(){const done=Object.values(ans).filter(a=>a.label).length;document.getElementById("prog").textContent=done+" / "+total()+" etiket"}
function exportCsv(){if(!who.value){alert("Önce adınızı yazın");return}
 let rows=["item,noun_index,word,cls,in_vocab,label,error_type,box,substitute,substitute_ok,annotator"];
 DATA.forEach(d=>d.nouns.forEach((n,j)=>{const k=d.item+"|"+j;const rec=ans[k]||{};
  const box=rec.box?rec.box.map(v=>v.toFixed(4)).join(":"):"";
  rows.push([d.item,j,n.word,n.cls,n.in_vocab?1:0,rec.label||"",rec.error_type||"",box,n.ask_substitute||"",rec.substitute_ok||"",who.value].join(","))}));
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows.join("\\n")],{type:"text/csv"}));
 a.download="labels_"+NAME+"_"+who.value.replace(/\\W+/g,"_")+".csv";a.click()}
render();
</script></html>"""
out = config.ROOT / "annotation" / a.name
out.mkdir(parents=True, exist_ok=True)
(out / "annotate.html").write_text(html.replace("__NAME__", a.name).replace("__DATA__", json.dumps(payload)))
meta = dict(systems=a.systems, split=a.split, decoder=a.decoder, seed=a.seed,
            items=[dict(item=it["item"], image=it["image"], caption=it["caption"]) for it in items], system_captions=key)
if a.pilot:
    meta.update(mode="lcr_pilot", protocol="docs/PILOT_PROTOKOL.md", records=dict(s4="A_B_lcr4.json", s2="A_B_lcr2.json"),
                s4_bundle_id=s4_bundle, sizes=dict(n_random=a.n_random, n_s2_keep=a.n_s2_keep, n_elsewhere=a.n_elsewhere,
                                                    n_keep=a.n_keep),
                population={g: dict(in_group=n, sampled=k) for g, (n, k) in population.items()},
                focus=focus_out, focus_unmatched=unmatched)
cache.save_json(meta, out / "key.json")
if a.pilot:
    print({g: v["sampled"] for g, v in meta["population"].items()}, f"unmatched {unmatched}")
print(f"{len(items)} unique (image, caption) items, {sum(len(i['nouns']) for i in items)} nouns -> {out}/annotate.html")
