import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

REPO_ID = "Samira-Hassan/bert-ner-production-model"

tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
model = AutoModelForTokenClassification.from_pretrained(REPO_ID)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_entities(text):
    if not text.strip():
        return {"text": "", "entities": []}
    results = ner_pipeline(text)
    entities = []
    for ent in results:
        entities.append({
            "entity": ent["entity_group"],
            "word": ent["word"],
            "start": ent["start"],
            "end": ent["end"],
            "score": round(float(ent["score"]), 4)
        })
    return {"text": text, "entities": entities}

demo = gr.Interface(
    fn=extract_entities,
    inputs=gr.Textbox(lines=3, placeholder="اكتبي جملة هنا... مثلاً: Elon Musk visited Cairo.", label="Input Sentence"),
    outputs=gr.HighlightedText(label="Extracted Entities"),
    title="🏷️ BERT Named Entity Recognition (NER)",
    description="مشروع تخرج / تبريز ذكاء اصطناعي لنشر موديل BERT استخراج الكيانات المسماة.",
    examples=[
        ["SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT ."],
        ["Apple is looking at buying U.K. startup for $1 billion."],
        ["Ahmed works at Google in London."]
    ]
)

if __name__ == "__main__":
    demo.launch()
