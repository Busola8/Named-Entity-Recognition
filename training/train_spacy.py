import random, plac, spacy, os
from spacy.util import minibatch, compounding
import json

TRAIN_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_data.jsonl')
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'ner_model')

def load_data(path):
    examples = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            obj = json.loads(line)
            text = obj['text']
            ents = obj['entities']
            # spaCy expects (start, end, label)
            examples.append((text, {"entities": ents}))
    return examples

def main(n_iter=30):
    examples = load_data(TRAIN_DATA_PATH)
    # create blank model
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    # add labels
    for _, ann in examples:
        for ent in ann.get("entities"):
            ner.add_label(ent[2])
    # begin training
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(examples)
        losses = {}
        batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
        print("Iteration", itn, "losses", losses)
    # save model
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    nlp.to_disk(MODEL_OUTPUT_DIR)
    print("Saved model to", MODEL_OUTPUT_DIR)

if __name__ == '__main__':
    main()
