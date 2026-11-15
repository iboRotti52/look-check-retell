"""Build the blinded annotation page for the error pilot (or any later human evaluation).

    python scripts/07_make_annotation.py --name pilot --n 100 --systems draft
    python scripts/07_make_annotation.py --name pilot2 --n 100 --systems draft ours b0plus

Writes annotation/<name>/annotate.html (self-contained, works offline) and
annotation/<name>/key.json (which system produced which caption -- do NOT open
it before all labels are in). Each unique (image, caption) pair appears once;
items are shuffled so that versions of one image are not adjacent.
"""
import argparse
import base64
import io
import json

import nltk
import numpy as np

import _common  # noqa: F401
from lcr import cache, claims, config, data

NOT_OBJECTS = set("front top air background side middle back end edge way lot bunch pair kind day night "
                  "time picture photo view trick game part piece distance area set".split())

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="pilot")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--split", default="validation")
ap.add_argument("--decoder", default="A")
ap.add_argument("--systems", nargs="+", default=["draft"])
ap.add_argument("--seed", type=int, default=config.SEED)
a = ap.parse_args()
assert a.split != "test" or a.name.startswith("final"), "test images only for the final evaluation"

rows = {r["id"]: r for r in cache.load_json(config.CACHE / "pipeline" / f"{a.decoder}_{a.split}.json")}
rng = np.random.default_rng(a.seed)
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
cinsiyet ve ya&#351; de&#287;erlendirilmez. Eylemler, renkler ve say&#305;lar de&#287;erlendirilmez.</p>
<div class="bar">Etiketleyen: <input id="who" placeholder="ad"> &nbsp; <span id="prog"></span> &nbsp;
<button onclick="exportCsv()">CSV indir</button> <small>Yan&#305;tlar taray&#305;c&#305;da otomatik saklan&#305;r.</small></div>
<div id="list"></div>
<script>
const NAME="__NAME__", DATA=__DATA__, OPTS=[["correct","doğru"],["incorrect","yanlış"],["uncertain","emin değil"],["na","nesne değil"]];
let ans=JSON.parse(localStorage.getItem("lcr-"+NAME)||"{}");
const who=document.getElementById("who"); who.value=localStorage.getItem("lcr-who")||""; who.oninput=()=>localStorage.setItem("lcr-who",who.value);
function render(){const L=document.getElementById("list");L.innerHTML="";DATA.forEach(d=>{const c=document.createElement("div");c.className="card";
 let h='<img src="data:image/jpeg;base64,'+d.img+'"><div><div class="cap">'+d.caption+'</div><small>'+d.item+'</small>';
 d.nouns.forEach((n,j)=>{const k=d.item+"|"+j;h+='<div class="noun '+(ans[k]?"":"todo")+'" id="n-'+k+'"><b><u>'+n.word+'</u></b>';
  OPTS.forEach(o=>{h+='<label><input type="radio" name="'+k+'" value="'+o[0]+'" '+(ans[k]===o[0]?"checked":"")+' onchange="setA(\\''+k+'\\',this.value)">'+o[1]+'</label>'});h+='</div>'});
 if(!d.nouns.length)h+='<small>(nesne ismi yok)</small>';c.innerHTML=h+'</div>';L.appendChild(c)});prog()}
function setA(k,v){ans[k]=v;localStorage.setItem("lcr-"+NAME,JSON.stringify(ans));document.getElementById("n-"+k).classList.remove("todo");prog()}
function total(){return DATA.reduce((s,d)=>s+d.nouns.length,0)}
function prog(){document.getElementById("prog").textContent=Object.keys(ans).length+" / "+total()+" etiket"}
function exportCsv(){if(!who.value){alert("Önce adınızı yazın");return}
 let rows=["item,noun_index,word,cls,in_vocab,label,annotator"];
 DATA.forEach(d=>d.nouns.forEach((n,j)=>rows.push([d.item,j,n.word,n.cls,n.in_vocab?1:0,ans[d.item+"|"+j]||"",who.value].join(","))));
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows.join("\\n")],{type:"text/csv"}));
 a.download="labels_"+NAME+"_"+who.value.replace(/\\W+/g,"_")+".csv";a.click()}
render();
</script></html>"""
out = config.ROOT / "annotation" / a.name
out.mkdir(parents=True, exist_ok=True)
(out / "annotate.html").write_text(html.replace("__NAME__", a.name).replace("__DATA__", json.dumps(payload)))
cache.save_json(dict(systems=a.systems, split=a.split, decoder=a.decoder, seed=a.seed,
                     items=[dict(item=it["item"], image=it["image"], caption=it["caption"]) for it in items],
                     system_captions=key), out / "key.json")
print(f"{len(items)} unique (image, caption) items, {sum(len(i['nouns']) for i in items)} nouns -> {out}/annotate.html")
