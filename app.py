import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_ID = "Samira-Hassan/bert-ner-production-model"

print("Loading model and tokenizer from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
ner_pipeline = pipeline(
    "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
)

# ------------------------------------------------------------------ #
# Entity metadata (single source of truth for colors + names)
# ------------------------------------------------------------------ #
ENTITIES = {
    "PER":  {"name": "Person",       "color": "#3b82f6", "bg": "#dbeafe", "icon": "👤"},
    "ORG":  {"name": "Organization", "color": "#10b981", "bg": "#d1fae5", "icon": "🏢"},
    "LOC":  {"name": "Location",     "color": "#8b5cf6", "bg": "#ede9fe", "icon": "📍"},
    "MISC": {"name": "Miscellaneous","color": "#f59e0b", "bg": "#fef3c7", "icon": "🏷️"},
}
COLOR_MAP = {k: v["bg"] for k, v in ENTITIES.items()}

EMPTY_TABLE = []


def stats_html(counts=None):
    """Four summary cards, one per entity type."""
    counts = counts or {}
    cards = ""
    for key, meta in ENTITIES.items():
        n = counts.get(key, 0)
        cards += f"""
        <div class="stat-card" style="--accent:{meta['color']}; --soft:{meta['bg']}">
            <div class="stat-icon">{meta['icon']}</div>
            <div class="stat-body">
                <div class="stat-count">{n}</div>
                <div class="stat-name">{meta['name']}</div>
            </div>
        </div>"""
    return f'<div class="stat-grid">{cards}</div>'


def predict_ner(text):
    if not text or not text.strip():
        return [], stats_html(), EMPTY_TABLE

    results = ner_pipeline(text)

    # Build segments for HighlightedText: [(text, label|None), ...]
    segments, cursor = [], 0
    counts, rows = {}, []
    for ent in results:
        start, end = ent["start"], ent["end"]
        label = ent["entity_group"]
        if start > cursor:
            segments.append((text[cursor:start], None))
        segments.append((text[start:end], label))
        cursor = end

        counts[label] = counts.get(label, 0) + 1
        rows.append([
            text[start:end],
            f"{ENTITIES.get(label, {}).get('icon', '')} {ENTITIES.get(label, {}).get('name', label)}",
            f"{float(ent['score']) * 100:.1f}%",
        ])
    if cursor < len(text):
        segments.append((text[cursor:], None))

    return segments, stats_html(counts), rows


def clear_all():
    return "", [], stats_html(), EMPTY_TABLE


# ------------------------------------------------------------------ #
# Styling
# ------------------------------------------------------------------ #
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root { --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --brand:#4f46e5; }

.gradio-container {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
    max-width: 1180px !important;
    margin: 0 auto !important;
    background: linear-gradient(180deg, #f1f5f9 0%, #ffffff 340px) !important;
}

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #3730a3 55%, #4f46e5 100%);
    border-radius: 20px;
    padding: 38px 40px;
    margin: 18px 0 22px;
    color: #fff;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute; right: -60px; top: -60px;
    width: 260px; height: 260px; border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,.18), transparent 70%);
}
.hero h1 {
    margin: 0 0 8px; font-size: 2rem; font-weight: 800;
    letter-spacing: -0.02em; color: #fff;
}
.hero p { margin: 0; color: #c7d2fe; font-size: 1.02rem; max-width: 640px; line-height: 1.6; }
.hero-badges { margin-top: 18px; display: flex; gap: 8px; flex-wrap: wrap; }
.hero-badge {
    background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22);
    padding: 5px 12px; border-radius: 999px; font-size: .8rem; font-weight: 600; color: #e0e7ff;
}

/* ---------- Panels ---------- */
.panel {
    background: #fff !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    padding: 18px !important;
    box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -12px rgba(15,23,42,.10) !important;
}
.section-title {
    font-weight: 700; font-size: .95rem; color: var(--ink); margin-bottom: 4px;
}

/* ---------- Buttons ---------- */
#run-btn {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    border: none !important; color: #fff !important;
    font-weight: 700 !important; border-radius: 12px !important;
    box-shadow: 0 6px 16px -6px rgba(79,70,229,.6) !important;
    transition: transform .15s ease, box-shadow .15s ease;
}
#run-btn:hover { transform: translateY(-1px); box-shadow: 0 10px 20px -8px rgba(79,70,229,.7) !important; }
#clear-btn { border-radius: 12px !important; font-weight: 600 !important; }

/* ---------- Stat cards ---------- */
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.stat-card {
    display: flex; align-items: center; gap: 12px;
    background: var(--soft); border-left: 4px solid var(--accent);
    border-radius: 12px; padding: 12px 14px;
}
.stat-icon { font-size: 1.4rem; }
.stat-count { font-size: 1.5rem; font-weight: 800; color: var(--accent); line-height: 1; }
.stat-name  { font-size: .78rem; color: #334155; font-weight: 600; margin-top: 3px; }
@media (max-width: 720px) { .stat-grid { grid-template-columns: repeat(2, 1fr); } .hero { padding: 26px 22px; } }

/* ---------- Footer ---------- */
.footer { text-align: center; color: var(--muted); font-size: .82rem; padding: 22px 0 8px; }
.footer code { background: #eef2ff; color: #4338ca; padding: 2px 8px; border-radius: 6px; }
footer { display: none !important; }
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "system-ui", "sans-serif"],
)

HERO = f"""
<div class="hero">
    <h1>Named Entity Recognition</h1>
    <p>Paste any English text and the model will find the people, organizations,
       locations and other named entities inside it.</p>
    <div class="hero-badges">
        <span class="hero-badge">BERT fine-tuned</span>
        <span class="hero-badge">CoNLL-2003</span>
        <span class="hero-badge">4 entity types</span>
    </div>
</div>
"""

FOOTER = f'<div class="footer">Model: <code>{MODEL_ID}</code></div>'

# ------------------------------------------------------------------ #
# App
# ------------------------------------------------------------------ #
with gr.Blocks(theme=theme, css=CSS, title="NER · BERT") as demo:
    gr.HTML(HERO)

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes="panel"):
            gr.HTML('<div class="section-title">Your text</div>')
            input_text = gr.Textbox(
                show_label=False,
                placeholder="Example: Tim Cook visited Apple headquarters in California.",
                lines=9,
            )
            with gr.Row():
                clear_btn = gr.Button("Clear", elem_id="clear-btn", scale=1)
                submit_btn = gr.Button("Extract entities", variant="primary",
                                       elem_id="run-btn", scale=2)

            gr.Examples(
                examples=[
                    ["SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT ."],
                    ["Elon Musk announced that Tesla will open a new factory in Germany."],
                    ["Reuters reported that the United Nations meeting in New York ended yesterday."],
                ],
                inputs=input_text,
                label="Try an example",
            )

        with gr.Column(scale=6, elem_classes="panel"):
            gr.HTML('<div class="section-title">Results</div>')
            stats = gr.HTML(stats_html())
            output_highlight = gr.HighlightedText(
                show_label=False,
                combine_adjacent=False,
                show_legend=True,
                color_map=COLOR_MAP,
            )
            table = gr.Dataframe(
                headers=["Entity", "Type", "Confidence"],
                datatype=["str", "str", "str"],
                interactive=False,
                wrap=True,
                label="Detected entities",
            )

    gr.HTML(FOOTER)

    outputs = [output_highlight, stats, table]
    submit_btn.click(fn=predict_ner, inputs=input_text, outputs=outputs)
    input_text.submit(fn=predict_ner, inputs=input_text, outputs=outputs)
    clear_btn.click(fn=clear_all, outputs=[input_text, *outputs])

if __name__ == "__main__":
    demo.launch()
