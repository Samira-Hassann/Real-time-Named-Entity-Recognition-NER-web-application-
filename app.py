
import gradio as gr
import html

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)

# ==========================================
# 1. Model Configuration
# ==========================================

MODEL_ID = "Samira-Hassan/bert-ner-production-model"

# ==========================================
# 2. Load Model and Tokenizer
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_ID
)

ner_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=-1
)

# ==========================================
# 3. Entity Colors
# ==========================================

ENTITY_COLORS = {
    "PER": "#DCEBFF",
    "ORG": "#DDF5E5",
    "LOC": "#FFE8C7",
    "MISC": "#F0DFFF"
}


# ==========================================
# 4. Prediction Function
# ==========================================

def predict_entities(text):

    if not text or not text.strip():
        return (
            "<p>Please enter some text.</p>",
            []
        )

    predictions = ner_pipeline(
        text,
        truncation=True
    )

    # Sort entities by their positions
    predictions = sorted(
        predictions,
        key=lambda x: x["start"]
    )

    highlighted_text = ""
    previous_end = 0
    results = []

    for entity in predictions:

        start = entity["start"]
        end = entity["end"]

        label = entity["entity_group"]
        word = entity["word"]
        confidence = float(entity["score"])

        # Avoid overlapping spans
        if start < previous_end:
            continue

        # Add normal text
        highlighted_text += html.escape(
            text[previous_end:start]
        )

        color = ENTITY_COLORS.get(
            label,
            "#EEEEEE"
        )

        # Add highlighted entity
        highlighted_text += f"""
        <span style="
            background-color: {color};
            padding: 4px 7px;
            border-radius: 5px;
            margin: 2px;
            display: inline-block;
        ">
            <b>{html.escape(text[start:end])}</b>
            <small>[{html.escape(label)}]</small>
        </span>
        """

        results.append({
            "Entity": text[start:end],
            "Label": label,
            "Confidence": round(confidence, 4),
            "Start": start,
            "End": end
        })

        previous_end = end

    # Add remaining text
    highlighted_text += html.escape(
        text[previous_end:]
    )

    highlighted_html = f"""
    <div style="
        font-size: 17px;
        line-height: 2.5;
        padding: 15px;
        border-radius: 10px;
    ">
        {highlighted_text}
    </div>
    """

    return highlighted_html, results


# ==========================================
# 5. Gradio Interface
# ==========================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="BERT NER Deployment"
) as demo:

    gr.Markdown("""
    # 🧠 BERT Named Entity Recognition

    Fine-tuned BERT model for identifying named entities
    in English text.

    **Entity Types:** PER | ORG | LOC | MISC
    """)

    input_text = gr.Textbox(
        label="Input Text",
        placeholder="Enter your text here...",
        lines=5
    )

    predict_button = gr.Button(
        "Recognize Entities",
        variant="primary"
    )

    gr.Markdown("### 🎨 Highlighted Entities")

    highlighted_output = gr.HTML()

    gr.Markdown("### 📊 Prediction Results")

    results_output = gr.JSON()

    gr.Examples(
        examples=[
            ["Barack Obama visited London."],
            ["Apple hired John in New York."],
            ["Microsoft is an American company."]
        ],
        inputs=input_text
    )

    predict_button.click(
        fn=predict_entities,
        inputs=input_text,
        outputs=[
            highlighted_output,
            results_output
        ]
    )

demo.launch()
