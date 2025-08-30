import streamlit as st, os
from app.inference import load_model, extract_entities

st.set_page_config(page_title='NER Auto-Fill Demo', layout='wide')
st.title('Named Entity Recognition — Auto-Fill Demo')

nlp = load_model()

st.sidebar.header('Input')
input_option = st.sidebar.radio('Source', ['Paste text', 'Upload file (txt/pdf)'])
text = ''

if input_option == 'Paste text':
    text = st.text_area('Paste document text here:', height=200)
else:
    uploaded = st.file_uploader('Upload a .txt file', type=['txt'])
    if uploaded is not None:
        text = uploaded.read().decode('utf8')

if st.button('Extract entities'):
    if not text:
        st.warning('Please provide text or upload a file.')
    else:
        ents = extract_entities(text, nlp=nlp)
        st.subheader('Entities found')
        for ent, label in ents:
            st.write(f'- **{label}**: {ent}')
        # Simple auto-fill form example
        st.subheader('Auto-filled form (example fields)')
        fields = {'PERSON':'Name', 'DATE':'Date', 'GPE':'Location', 'ORG':'Organization', 'EMAIL':'Email', 'PHONE':'Phone', 'MONEY':'Amount', 'PASSPORT':'Passport', 'ADDRESS':'Address', 'ACCOUNT':'Account'}
        filled = {}
        for ent, label in ents:
            if label in fields and fields[label] not in filled:
                filled[fields[label]] = ent
        for k,v in filled.items():
            st.text_input(k, value=v)
