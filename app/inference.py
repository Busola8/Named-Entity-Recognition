import os, spacy, re
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'ner_model')
def load_model():
    if os.path.exists(MODEL_DIR):
        print('Loading custom NER model...')
        nlp = spacy.load(MODEL_DIR)
    else:
        print('Loading spaCy en_core_web_sm fallback model...')
        try:
            nlp = spacy.load('en_core_web_sm')
        except Exception:
            import subprocess, sys
            subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
            nlp = spacy.load('en_core_web_sm')
    return nlp

def extract_entities(text, nlp=None):
    if nlp is None:
        nlp = load_model()
    doc = nlp(text)
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    # simple email/phone regex fallback
    emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    phones = re.findall(r'\+?\d[\d\s-]{7,}\d', text)
    if emails:
        ents.extend([(e, 'EMAIL') for e in emails if e not in [t for t,_ in ents]])
    if phones:
        ents.extend([(p, 'PHONE') for p in phones if p not in [t for t,_ in ents]])
    return ents

if __name__ == '__main__':
    import sys
    txt = ' '.join(sys.argv[1:]) if len(sys.argv)>1 else 'John Smith, email john@example.com, Lagos.'
    nlp = load_model()
    print(extract_entities(txt, nlp=nlp))
