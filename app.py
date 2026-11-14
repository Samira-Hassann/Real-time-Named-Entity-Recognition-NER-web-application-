import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_ID = "Samira-Hassan/bert-ner-production-model"

print("Loading model and tokenizer from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def predict_ner(text):
    if not text.strip():
        return {"text": text, "entities": []}
    
    results = ner_pipeline(text)
    
    entities = []
    for ent in results:
        entities.append({
            "entity": ent["entity_group"],
            "start": ent["start"],
            "end": ent["end"]
        })
    
    return {"text": text, "entities": entities}

# Build the Gradio web interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏷️ CoNLL-2003 Named Entity Recognition (BERT)")
    gr.Markdown(f"Production NER web application powered by your fine-tuned BERT model (`{MODEL_ID}`).")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Input English Text",
                placeholder="Example: Tim Cook visited Apple headquarters in California.",
                lines=4
            )
            submit_btn = gr.Button("Extract Entities", variant="primary")
            
        with gr.Column():
            output_highlight = gr.HighlightedText(
                label="Extracted Entities",
                combine_adjacent=True,
                show_legend=True,
                color_map={
                    "PER": "#60a5fa", 
                    "ORG": "#4ade80", 
                    "LOC": "#c084fc", 
                    "MISC": "#fb923c"
                }
            )
            
    gr.Examples(
        examples=[
            ["SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT ."],
            ["Elon Musk announced that Tesla will open a new factory in Germany."],
            ["Reuters reported that the United Nations meeting in New York ended yesterday."]
        ],
        inputs=input_text
    )
    
    submit_btn.click(fn=predict_ner, inputs=input_text, outputs=output_highlight)

if __name__ == "__main__":
    demo.launch()
