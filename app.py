import csv
import html
import json
import os
import tempfile
import time

import gradio as gr
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

MODEL_ID = "Samira-Hassan/bert-ner-production-model"

print("Loading model and tokenizer from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
ner_pipeline = pipeline(
    "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
)

# ---------------------------------------------------------------------------
# Entity metadata
# ---------------------------------------------------------------------------
ENT = {
    "PER":  {"name": "Person",        "c": "#4cc9ff", "bg": "rgba(76,201,255,.18)"},
    "ORG":  {"name": "Organization",  "c": "#3ee6a8", "bg": "rgba(62,230,168,.18)"},
    "LOC":  {"name": "Location",      "c": "#b197fc", "bg": "rgba(177,151,252,.20)"},
    "MISC": {"name": "Miscellaneous", "c": "#ffb454", "bg": "rgba(255,180,84,.20)"},
}
FALLBACK = {"name": "Other", "c": "#94a3b8", "bg": "rgba(148,163,184,.18)"}


def meta(label):
    return ENT.get(label, FALLBACK)


def esc(s):
    return html.escape(str(s))


# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------
def skeleton(title, hint):
    return f"""
    <div class="empty">
        <div class="sk">
            <span class="sk-line" style="width:92%"></span>
            <span class="sk-line"><i class="sk-hl" style="--c:#4cc9ff;width:22%"></i><i style="width:46%"></i><i class="sk-hl" style="--c:#b197fc;width:18%"></i></span>
            <span class="sk-line"><i style="width:34%"></i><i class="sk-hl" style="--c:#3ee6a8;width:26%"></i><i style="width:28%"></i></span>
            <span class="sk-line" style="width:64%"></span>
        </div>
        <div class="empty-title">{esc(title)}</div>
        <div class="empty-hint">{esc(hint)}</div>
    </div>"""


def render_summary(ents, ms, n_chars):
    avg = sum(float(e["score"]) for e in ents) / len(ents) * 100
    counts = {}
    for e in ents:
        counts[e["entity_group"]] = counts.get(e["entity_group"], 0) + 1

    metrics = f"""
    <div class="metrics">
        <div class="metric"><div class="m-val">{len(ents)}</div><div class="m-lbl">Entities</div></div>
        <div class="metric"><div class="m-val">{avg:.0f}<span>%</span></div><div class="m-lbl">Avg. confidence</div></div>
        <div class="metric"><div class="m-val">{ms:.0f}<span>ms</span></div><div class="m-lbl">Inference</div></div>
        <div class="metric"><div class="m-val">{n_chars}</div><div class="m-lbl">Characters</div></div>
    </div>"""

    segs, legend = "", ""
    for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        m = meta(label)
        pct = n / len(ents) * 100
        segs += (
            f'<div class="seg" style="width:{pct:.2f}%;--c:{m["c"]}" '
            f'title="{esc(m["name"])}: {n}"></div>'
        )
        legend += (
            f'<span class="lg" style="--c:{m["c"]}"><i></i>{esc(m["name"])}'
            f'<b>{n}</b></span>'
        )
    return f"""{metrics}
    <div class="dist"><div class="bar">{segs}</div><div class="legend">{legend}</div></div>"""


def render_text(text, ents):
    out, cur, i = [], 0, 0
    for e in ents:
        s, t = e["start"], e["end"]
        if s < cur:
            continue
        out.append(esc(text[cur:s]))
        m = meta(e["entity_group"])
        out.append(
            f'<mark class="ent" style="--c:{m["c"]};--tint:{m["bg"]};--i:{i}" '
            f'title="{esc(m["name"])} · {float(e["score"]) * 100:.1f}%">'
            f'{esc(text[s:t])}<span class="tag">{esc(e["entity_group"])}</span></mark>'
        )
        cur, i = t, i + 1
    out.append(esc(text[cur:]))
    return f'<div class="annotated">{"".join(out)}</div>'


def render_table(text, ents):
    rows = ""
    for e in ents:
        m = meta(e["entity_group"])
        conf = float(e["score"]) * 100
        rows += f"""
        <tr>
            <td class="t-ent">{esc(text[e['start']:e['end']])}</td>
            <td><span class="pill" style="--c:{m['c']};--tint:{m['bg']}">{esc(m['name'])}</span></td>
            <td class="t-conf">
                <div class="cbar"><div style="width:{conf:.1f}%;--c:{m['c']}"></div></div>
                <span>{conf:.1f}%</span>
            </td>
        </tr>"""
    return f"""
    <div class="table-wrap">
    <table class="ent-table">
        <thead><tr><th>Entity</th><th>Type</th><th>Confidence</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    </div>"""


def render_results(text, ents, ms):
    if not ents:
        return skeleton(
            "No entities found",
            "Try lowering the minimum confidence or enabling more types in Settings.",
        )
    return (
        render_summary(ents, ms, len(text))
        + '<div class="sec-title">Annotated text</div>'
        + render_text(text, ents)
        + '<div class="sec-title">All entities</div>'
        + render_table(text, ents)
    )


IDLE = skeleton("Your results will appear here", "Write or pick a sentence, then press Extract entities.")


# ---------------------------------------------------------------------------
# Inference + export
# ---------------------------------------------------------------------------
def export_files(text, ents):
    tmp = tempfile.mkdtemp()
    data = [
        {
            "text": text[e["start"]:e["end"]],
            "type": e["entity_group"],
            "confidence": round(float(e["score"]), 4),
            "start": int(e["start"]),
            "end": int(e["end"]),
        }
        for e in ents
    ]
    jpath = os.path.join(tmp, "entities.json")
    cpath = os.path.join(tmp, "entities.csv")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with open(cpath, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "type", "confidence", "start", "end"])
        w.writeheader()
        w.writerows(data)
    return [jpath, cpath]


def predict(text, threshold, types):
    if not text or not text.strip():
        return IDLE, gr.update(value=None, visible=False)

    t0 = time.perf_counter()
    raw = ner_pipeline(text)
    ms = (time.perf_counter() - t0) * 1000

    allowed = set(types or [])
    ents = [
        e for e in raw
        if float(e["score"]) >= threshold and e["entity_group"] in allowed
    ]
    files = export_files(text, ents) if ents else None
    return render_results(text, ents, ms), gr.update(value=files, visible=bool(files))


def reset():
    return "", IDLE, gr.update(value=None, visible=False)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap');

:root, .dark, .gradio-container {
    --ink:#f3f4ff; --ink-2:#c5c8e6; --muted:#8b8fb5;
    --line:rgba(255,255,255,.09); --glass:rgba(255,255,255,.045);
    --violet:#7c5cff; --cyan:#22d3ee; --pink:#ff5ca8;

    /* neutralise Gradio's own surfaces so our glass shows through */
    --body-background-fill:#07081a;
    --background-fill-primary:transparent;
    --background-fill-secondary:rgba(255,255,255,.04);
    --block-background-fill:transparent;
    --block-border-color:transparent;
    --border-color-primary:rgba(255,255,255,.12);
    --input-background-fill:rgba(8,9,28,.55);
    --input-border-color:rgba(255,255,255,.12);
    --body-text-color:#f3f4ff;
    --block-label-text-color:#c5c8e6;
    --block-title-text-color:#c5c8e6;
    --checkbox-label-background-fill:rgba(255,255,255,.05);
    --checkbox-label-background-fill-hover:rgba(255,255,255,.09);
    --checkbox-label-text-color:#f3f4ff;
    --checkbox-background-color-selected:#7c5cff;
    --slider-color:#7c5cff;
}

html, body, gradio-app { background:#07081a !important; }

/* ---------- Aurora backdrop ---------- */
body::before {
    content:""; position:fixed; inset:-20%; z-index:0; pointer-events:none;
    background:
        radial-gradient(40% 35% at 18% 20%, rgba(124,92,255,.38), transparent 70%),
        radial-gradient(38% 32% at 82% 12%, rgba(34,211,238,.26), transparent 70%),
        radial-gradient(36% 34% at 70% 85%, rgba(255,92,168,.22), transparent 70%),
        radial-gradient(30% 30% at 12% 80%, rgba(62,230,168,.14), transparent 70%);
    filter:blur(30px);
    animation:drift 28s ease-in-out infinite alternate;
}
body::after {
    content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
    background-image:radial-gradient(rgba(255,255,255,.05) 1px, transparent 1px);
    background-size:26px 26px;
    mask-image:radial-gradient(70% 60% at 50% 30%, #000, transparent);
    -webkit-mask-image:radial-gradient(70% 60% at 50% 30%, #000, transparent);
}
@keyframes drift {
    from { transform:translate3d(-2%, -1%, 0) scale(1); }
    to   { transform:translate3d(3%, 2%, 0) scale(1.08); }
}

.gradio-container {
    position:relative; z-index:1;
    font-family:'DM Sans', system-ui, sans-serif !important;
    max-width:1200px !important; margin:0 auto !important;
    background:transparent !important; color:var(--ink);
    padding:0 18px !important;
}
footer { display:none !important; }

/* ---------- Hero ---------- */
.nav { display:flex; justify-content:space-between; align-items:center; padding:26px 0 0; }
.brand { display:flex; align-items:center; gap:12px; font:700 1rem 'Sora', sans-serif; color:var(--ink); }
.logo {
    width:36px; height:36px; border-radius:11px; display:grid; place-items:center;
    background:linear-gradient(135deg, var(--violet), var(--cyan));
    box-shadow:0 8px 24px -6px rgba(124,92,255,.7);
    font:700 .8rem 'Sora', sans-serif; color:#fff;
}
.model-chip {
    padding:7px 14px; border-radius:999px; font-size:.78rem; color:var(--ink-2);
    background:var(--glass); border:1px solid var(--line); backdrop-filter:blur(12px);
}
.model-chip i { display:inline-block; width:7px; height:7px; border-radius:50%; background:#3ee6a8;
    margin-right:8px; box-shadow:0 0 10px #3ee6a8; }

.hero { padding:56px 0 40px; max-width:860px; }
.hero h1 {
    margin:0; font-family:'Sora', sans-serif; font-weight:800;
    font-size:clamp(2.5rem, 6vw, 4.4rem); line-height:1.04; letter-spacing:-.035em;
    background:linear-gradient(100deg, #ffffff 10%, #c9bcff 45%, #7fe7ff 90%);
    -webkit-background-clip:text; background-clip:text; color:transparent;
}
.hero p { margin:20px 0 0; max-width:560px; font-size:1.1rem; line-height:1.65; color:var(--ink-2); }
.key { display:flex; gap:10px; flex-wrap:wrap; margin-top:26px; }
.key span {
    display:inline-flex; align-items:center; gap:8px; padding:7px 14px; border-radius:999px;
    font-size:.82rem; font-weight:600; color:var(--ink);
    background:var(--tint); border:1px solid var(--c); box-shadow:0 0 22px -10px var(--c);
}

/* ---------- Glass cards ---------- */
.glass {
    background:var(--glass) !important; border:1px solid var(--line) !important;
    border-radius:24px !important; padding:26px !important;
    backdrop-filter:blur(22px) saturate(140%); -webkit-backdrop-filter:blur(22px) saturate(140%);
    box-shadow:0 30px 80px -40px rgba(0,0,0,.8), inset 0 1px 0 rgba(255,255,255,.07) !important;
    gap:16px !important;
}
.glass .block, .glass .form, .glass .gap { background:transparent !important; border:none !important; box-shadow:none !important; }
.card-title { font:700 1.1rem 'Sora', sans-serif; color:var(--ink); letter-spacing:-.01em; }
.card-sub { font-size:.86rem; color:var(--muted); margin-top:4px; }

.gradio-container textarea {
    font-family:'Sora', sans-serif !important; font-weight:300 !important;
    font-size:1.05rem !important; line-height:1.7 !important; color:var(--ink) !important;
    background:rgba(6,7,24,.6) !important; border:1px solid var(--line) !important;
    border-radius:16px !important; padding:18px !important;
}
.gradio-container textarea::placeholder { color:#5f6390 !important; }
.gradio-container textarea:focus {
    border-color:rgba(124,92,255,.8) !important;
    box-shadow:0 0 0 4px rgba(124,92,255,.18), 0 0 40px -10px rgba(124,92,255,.6) !important;
}

/* ---------- Buttons ---------- */
#run-btn {
    min-height:50px; border:none !important; border-radius:14px !important;
    font:700 .98rem 'DM Sans', sans-serif !important; color:#fff !important;
    background:linear-gradient(110deg, var(--violet), #4f8bff 55%, var(--cyan)) !important;
    box-shadow:0 14px 34px -12px rgba(88,120,255,.9), inset 0 1px 0 rgba(255,255,255,.35) !important;
    transition:transform .18s ease, box-shadow .18s ease, filter .18s ease;
}
#run-btn:hover { transform:translateY(-2px); filter:brightness(1.08);
    box-shadow:0 20px 40px -12px rgba(88,120,255,1), inset 0 1px 0 rgba(255,255,255,.4) !important; }
#clear-btn {
    min-height:50px; border-radius:14px !important; font-weight:600 !important;
    background:var(--glass) !important; border:1px solid var(--line) !important; color:var(--ink-2) !important;
}
#clear-btn:hover { background:rgba(255,255,255,.09) !important; }

.try-label { font-size:.8rem; color:var(--muted); font-weight:600; margin-top:4px; }
.ex-btn {
    border-radius:12px !important; text-align:left !important; font-size:.86rem !important; font-weight:500 !important;
    background:rgba(255,255,255,.04) !important; border:1px solid var(--line) !important; color:var(--ink-2) !important;
    transition:border-color .15s ease, background .15s ease;
}
.ex-btn:hover { border-color:rgba(124,92,255,.7) !important; background:rgba(124,92,255,.12) !important; color:#fff !important; }

.gradio-container input[type=range] { accent-color:#7c5cff; }

/* ---------- Metrics ---------- */
.metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
.metric {
    padding:16px 18px; border-radius:16px; background:rgba(255,255,255,.05); border:1px solid var(--line);
}
.m-val {
    font:800 1.85rem 'Sora', sans-serif; letter-spacing:-.03em; line-height:1.05;
    background:linear-gradient(120deg, #fff, #a9c6ff); -webkit-background-clip:text; background-clip:text; color:transparent;
}
.m-val span { font-size:.85rem; font-weight:600; margin-left:3px; -webkit-text-fill-color:var(--muted); }
.m-lbl { font-size:.78rem; color:var(--muted); margin-top:8px; font-weight:500; }

.dist { margin-top:16px; }
.bar { display:flex; gap:3px; height:12px; }
.seg {
    height:100%; border-radius:99px; background:var(--c); box-shadow:0 0 18px -2px var(--c);
}
.legend { display:flex; gap:18px; flex-wrap:wrap; margin-top:12px; }
.lg { display:inline-flex; align-items:center; gap:8px; font-size:.82rem; color:var(--ink-2); }
.lg i { width:9px; height:9px; border-radius:50%; background:var(--c); box-shadow:0 0 10px var(--c); }
.lg b { color:var(--ink); margin-left:2px; }

.sec-title { font:700 .95rem 'Sora', sans-serif; color:var(--ink); margin:26px 0 12px; }

/* ---------- Annotated text ---------- */
.annotated {
    padding:26px 28px; border-radius:18px; background:rgba(6,7,24,.55); border:1px solid var(--line);
    font:300 1.22rem/2.5 'Sora', sans-serif; color:var(--ink-2);
    white-space:pre-wrap; word-break:break-word;
}
mark.ent {
    color:#fff; font-weight:600; padding:5px 9px; border-radius:9px; border:1px solid var(--c);
    background:linear-gradient(var(--tint), var(--tint)) no-repeat left center / 0% 100%;
    box-shadow:0 0 26px -8px var(--c);
    box-decoration-break:clone; -webkit-box-decoration-break:clone;
    animation:sweep .6s cubic-bezier(.2,.7,.2,1) forwards; animation-delay:calc(var(--i) * 110ms + 80ms);
}
@keyframes sweep { to { background-size:100% 100%; } }
.tag {
    margin-left:8px; padding:2px 7px; border-radius:6px; vertical-align:middle;
    font:700 .6rem 'DM Sans', sans-serif; letter-spacing:.04em; color:#07081a; background:var(--c);
}

/* ---------- Table ---------- */
.table-wrap { border:1px solid var(--line); border-radius:16px; overflow:hidden; background:rgba(255,255,255,.03); }
.ent-table { width:100%; border-collapse:collapse; font-size:.92rem; }
.ent-table th {
    text-align:left; padding:13px 18px; font-size:.76rem; font-weight:600; color:var(--muted);
    background:rgba(255,255,255,.04); border-bottom:1px solid var(--line);
}
.ent-table td { padding:14px 18px; border-bottom:1px solid var(--line); color:var(--ink); }
.ent-table tr:last-child td { border-bottom:none; }
.ent-table tbody tr:hover td { background:rgba(255,255,255,.04); }
.t-ent { font-weight:600; }
.pill {
    padding:4px 12px; border-radius:999px; font-size:.78rem; font-weight:600; white-space:nowrap;
    color:var(--c); background:var(--tint); border:1px solid var(--c);
}
.t-conf { display:flex; align-items:center; gap:12px; min-width:190px; }
.t-conf span { font-weight:600; font-size:.84rem; color:var(--ink-2); width:48px; }
.cbar { flex:1; height:6px; border-radius:99px; background:rgba(255,255,255,.08); overflow:hidden; }
.cbar div { height:100%; border-radius:99px; background:var(--c); box-shadow:0 0 12px var(--c); }

/* ---------- Empty / skeleton ---------- */
.empty { text-align:center; padding:50px 24px; border:1px dashed rgba(255,255,255,.14); border-radius:20px; }
.sk { display:flex; flex-direction:column; gap:12px; align-items:center; margin-bottom:26px; }
.sk-line { display:flex; gap:8px; width:70%; height:12px; }
.sk-line i { display:block; height:100%; border-radius:99px; background:rgba(255,255,255,.08); }
.sk-line:not(:has(i)) { background:rgba(255,255,255,.08); border-radius:99px; }
.sk-line i.sk-hl { background:var(--c); opacity:.75; box-shadow:0 0 16px -2px var(--c); }
.empty-title { font:700 1.05rem 'Sora', sans-serif; color:var(--ink); }
.empty-hint { font-size:.88rem; color:var(--muted); margin-top:6px; }

.foot { text-align:center; color:var(--muted); font-size:.8rem; padding:30px 0 28px; }
.foot code { background:var(--glass); border:1px solid var(--line); color:var(--ink-2); padding:3px 10px; border-radius:8px; }

@media (max-width:760px) {
    .metrics { grid-template-columns:repeat(2,1fr); }
    .glass { padding:18px !important; border-radius:20px !important; }
    .annotated { padding:18px; font-size:1.08rem; }
}
@media (prefers-reduced-motion:reduce) {
    body::before { animation:none; }
    mark.ent { animation:none; background-size:100% 100%; }
}
"""

FORCE_DARK = "() => { document.body.classList.add('dark'); }"

theme = gr.themes.Base(
    primary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("DM Sans"), "system-ui", "sans-serif"],
)

KEY = "".join(
    f'<span style="--c:{v["c"]};--tint:{v["bg"]}">{v["name"]}</span>' for v in ENT.values()
)
HERO = f"""
<div class="nav">
    <div class="brand"><div class="logo">NE</div>Entity Studio</div>
    <div class="model-chip"><i></i>BERT · CoNLL-2003</div>
</div>
<div class="hero">
    <h1>Find every name<br>in any text.</h1>
    <p>A fine-tuned BERT model that spots people, organizations, places and more
       in English text, in a fraction of a second.</p>
    <div class="key">{KEY}</div>
</div>
"""

EXAMPLES = [
    ("Soccer headline", "SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT ."),
    ("Tesla in Germany", "Elon Musk announced that Tesla will open a new factory in Germany."),
    ("UN in New York", "Reuters reported that the United Nations meeting in New York ended yesterday."),
]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
with gr.Blocks(theme=theme, css=CSS, js=FORCE_DARK, title="Entity Studio") as demo:
    gr.HTML(HERO)

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes="glass"):
            gr.HTML(
                '<div class="card-title">Your text</div>'
                '<div class="card-sub">Type or paste anything in English.</div>'
            )
            input_text = gr.Textbox(
                show_label=False,
                placeholder="Tim Cook visited Apple headquarters in California.",
                lines=9,
                max_lines=18,
            )
            with gr.Accordion("Settings", open=False):
                threshold = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="Minimum confidence",
                    info="Hide entities the model is less sure about.",
                )
                types = gr.CheckboxGroup(
                    choices=[(f"{v['name']} ({k})", k) for k, v in ENT.items()],
                    value=list(ENT.keys()),
                    label="Entity types",
                )
            with gr.Row():
                clear_btn = gr.Button("Clear", elem_id="clear-btn", scale=1)
                run_btn = gr.Button("Extract entities", elem_id="run-btn", scale=2)

            gr.HTML('<div class="try-label">Try an example</div>')
            ex_buttons = []
            for label, sample in EXAMPLES:
                b = gr.Button(label, elem_classes="ex-btn", size="sm")
                ex_buttons.append((b, sample))

        with gr.Column(scale=7, elem_classes="glass"):
            gr.HTML(
                '<div class="card-title">Results</div>'
                '<div class="card-sub">Hover any highlight to see how confident the model is.</div>'
            )
            results = gr.HTML(IDLE)
            export = gr.File(label="Download results (JSON + CSV)", visible=False)

    gr.HTML(f'<div class="foot">Powered by <code>{MODEL_ID}</code></div>')

    run_btn.click(predict, [input_text, threshold, types], [results, export])
    input_text.submit(predict, [input_text, threshold, types], [results, export])
    clear_btn.click(reset, None, [input_text, results, export])
    for b, sample in ex_buttons:
        b.click(lambda s=sample: s, None, input_text)

if __name__ == "__main__":
    demo.launch()
