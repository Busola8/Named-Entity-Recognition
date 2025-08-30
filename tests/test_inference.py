from app.inference import extract_entities, load_model
def test_extract_simple():
    nlp = load_model()
    ents = extract_entities('Alice Smith, email: alice@example.com, Lagos.', nlp=nlp)
    assert any(label=='EMAIL' for _, label in ents)
