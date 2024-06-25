from flask import Flask, request, render_template,jsonify
import joblib
from keras.models import load_model # type: ignore
import pandas as pd
import numpy as np
import tensorflow as tf
import spacy
import joblib
from sklearn.model_selection import train_test_split # type: ignore

app = Flask(__name__)

# Carregando o modelo e outros componentes necessários
spacy.prefer_gpu()
nlp = spacy.load("pt_core_news_sm")

# Carregar scaler e encoder treinados
scaler_path = './static/preprocessor_spacy2.pkl'
encoder_path = './static/label_encoder_spacy2.pkl'
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)

# Carregar o CSV com o mapeamento de 'codigo_solicitado' para suas descrições
csv_path = './static/cod_solic3.csv'
code_df = pd.read_csv(csv_path)
code_to_description = dict(zip(code_df['codigo_solicitado'], code_df['cod_solic_desc']))

# Carregar o arquivo de exemplos
examples_csv_path = './static/amostra_exemplo.csv'
examples_df = pd.read_csv(examples_csv_path)

# Carregar o modelo
model1 = load_model('./static/modelo_spacy2.h5')
def load_examples(start=0, count=50):
    print( examples_df.iloc[start:start + count].to_dict(orient='records'))
    return examples_df.iloc[start:start + count].to_dict(orient='records')


def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def preprocess_input(df, text_columns, scaler):
    df.fillna('', inplace=True)
    df.drop_duplicates(inplace=True)

    for col in text_columns:
        df[col] = df[col].apply(preprocess_text)

    print("df shape: ", df.shape)

    X_final = scaler.transform(df)
    print("X_final shape: ", X_final.shape)
    
    return X_final

def predict(df, model, text_columns, scaler, encoder, code_to_description, top_k=5):
    X_processed = preprocess_input(df, text_columns, scaler)
    predictions = model.predict(X_processed)
    top_k_indices = np.argsort(predictions, axis=1)[:, -top_k:][:, ::-1]
    top_k_probabilities = np.sort(predictions, axis=1)[:, -top_k:][:, ::-1]
    top_k_labels = [encoder.inverse_transform(indices) for indices in top_k_indices]
    top_k_descriptions = [[code_to_description[label] for label in labels] for labels in top_k_labels]
    return top_k_labels, top_k_probabilities, top_k_descriptions
    

@app.route('/', methods=['GET', 'POST'])
def index():
    examples = load_examples()
    
    if request.method == 'POST':
        abertura_id = request.form['abertura_id']
        modelo = request.form['modelo']
        problema = request.form['problema']
        tipo_equipamento = request.form['tipo_equipamento']

        # Criar o DataFrame com os dados de entrada
        data = {
            'abertura_id': [abertura_id],
            'modelo': [modelo],
            'problema': [problema],
            'tipo_equipamento': [tipo_equipamento]
        }
        df_input = pd.DataFrame(data)
      
        text_columns = ['problema', 'tipo_equipamento']
        
        # Fazer a previsão
        top_k_labels, top_k_probabilities, top_k_descriptions = predict(df_input, model1, text_columns, scaler, encoder, code_to_description, top_k=5)

        # Preparar os resultados para exibição
        results = []
        for labels, probabilities, descriptions in zip(top_k_labels, top_k_probabilities, top_k_descriptions):
            result = [{"label": label, "probability": prob, "description": desc} for label, prob, desc in zip(labels, probabilities, descriptions)]
            results.append(result)

        return render_template('index.html', results=results, examples=examples)

    return render_template('index.html', results=None, examples=examples)


@app.route('/more_examples', methods=['GET'])
def more_examples():
    start = int(request.args.get('start', 0))
    examples = load_examples(start=start, count=50)
    return jsonify(examples)

if __name__ == '__main__':
    app.run(debug=True)