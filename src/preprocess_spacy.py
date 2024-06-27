import pandas as pd
import numpy as np
import tensorflow as tf
import spacy

def preprocess_text(text):
    # Carregando o modelo e outros componentes necessários
    spacy.prefer_gpu()
    nlp = spacy.load("pt_core_news_sm")
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def preprocess_input(df, text_columns, scaler):
    df.fillna('', inplace=True)
    df.drop_duplicates(inplace=True)

    for col in text_columns:
        df[col] = df[col].apply(preprocess_text)

    X_final = scaler.transform(df)
    
    return X_final

