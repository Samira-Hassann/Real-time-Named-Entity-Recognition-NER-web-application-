import gradio as gr
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_ID = "./bert_ner_production_model"

print("Loading BERT NER model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
model.eval()

id2label = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC",
}


def predict_ner(text):
  if not text.strip():
    return []

  tokens = text.split()
  inputs = tokenizer(
      tokens,
      is_split_into_words=True,
      return_tensors="pt",
      padding=True,
      truncation=True,
  )

  with torch.no_grad():
    outputs = model(**inputs)

  predictions = torch.argmax(outputs.logits, dim=2)
  word_ids = inputs.word_ids(batch_index=0)
  predicted_tags = []
  previous_word_idx = None

  for word_idx in word_ids:
    if word_idx is None:
      continue
    if word_idx != previous_word_idx:
      token_idx = inputs.word_to_tokens(word_idx).start
      pred_id = predictions[0][token_idx].item()
      predicted_tags.append(id2label[pred_id])
      previous_word_idx = word_idx

  highlight_data = []
  for token, tag in zip(tokens, predicted_tags):
    entity_name = tag.split("-")[-1] if tag != "O" else None
    highlight_data.append((token, entity_name))

  return highlight_data


# Gradio Web UI Interface
demo = gr.Interface(
    fn=predict_ner,
    inputs=gr.Textbox(
        lines=3,
        placeholder=(
            "Enter an English sentence here (e.g., President Clinton visited"
            " Germany)..."
        ),
        label="Input Sentence",
    ),
    outputs=gr.HighlightedText(
        label="Extracted Named Entities",
        combine_adjacent=True,
        colors={
            "PER": "#D6EAF8",
            "ORG": "#D5F5E3",
            "LOC": "#E8DAEF",
            "MISC": "#FDEBD0",
        },
    ),
    title="🏷️ Named Entity Recognition (NER) with Fine-Tuned BERT",
    description=(
        "Real-time Named Entity Extraction using fine-tuned BERT model on"
        " CoNLL-2003 dataset."
    ),
    examples=[
        ["SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT ."],
        ["President Clinton visited Germany and met with officials in Berlin."],
    ],
)

if __name__ == "__main__":
  demo.launch()
