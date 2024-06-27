from keras.models import load_model # type: ignore
import joblib
from sklearn.model_selection import train_test_split # type: ignore


def load_files_spacy():
    # Carregar scaler e encoder treinados
    scaler_path = './static/preprocessor_spacy4.pkl'
    encoder_path = './static/label_encoder_spacy4.pkl'
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    # Carregar o modelo
    model = load_model('./static/modelo_spacy4.h5')

    return scaler, encoder, model