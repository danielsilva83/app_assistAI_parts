import pandas as pd
import joblib
from tensorflow.keras.models import load_model # type: ignore

def load_model_bert():
# Carregar scaler e encoder treinados
    scaler = joblib.load('./static/scaler17.pkl')
    encoder = joblib.load('./static/label_encoder17.pkl')
    model1 = load_model('./static/model17.h5')
    return model1, encoder, scaler