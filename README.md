# Named Entity Recognition (NER) Project — Portfolio

This project demonstrates a real-world NER pipeline oriented towards **KYC / form auto-fill** and **resume parsing**. It is intentionally _framework-light_ and does not use FastAPI; instead the demo uses Streamlit for an interactive UI.

## What you get
- `data/train_data.jsonl` — small synthetic annotated examples (spaCy format)
- `training/train_spacy.py` — lightweight spaCy training script (creates `models/ner_model`)
- `app/inference.py` — load model and extract entities (fallback to `en_core_web_sm` if no custom model)
- `app/streamlit_app.py` — demo UI to paste text / upload file, extract entities, and auto-fill example form fields
- `notebooks/ner_notebook.ipynb` — short walkthrough
- `requirements.txt` — minimal dependencies
- `tests/test_inference.py` — basic unit test for inference helper

## How to run locally
1. Create venv and install deps:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
2. Train the model (optional):
```bash
python training/train_spacy.py
```
3. Run Streamlit demo:
```bash
streamlit run app/streamlit_app.py
```
4. Try pasting sample text or uploading a .txt file. Click **Extract entities** and see the auto-filled fields.

