from flask import Flask, request, render_template
import torch
import joblib
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Carregando o modelo e outros componentes necessários
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Carregar scaler e encoder treinados
scaler = joblib.load('C:\\Users\\daniel.silva\\Documents\\cits\\desafio I\\resultado1\\scaler15.pkl')
encoder = joblib.load('C:\\Users\\daniel.silva\\Documents\\cits\\desafio I\\resultado1\\label_encoder15.pkl')

# Carregar o CSV com o mapeamento de 'codigo_solicitado' para suas descrições
csv_path = 'C:\\Users\\daniel.silva\\Documents\\cits\\desafio I\\amostra\\cod_solic3.csv'
code_df = pd.read_csv(csv_path)
code_to_description = dict(zip(code_df['codigo_solicitado'], code_df['cod_solic_desc']))

# Carregar o modelo
model1 = load_model('C:\\Users\\daniel.silva\\Documents\\cits\\desafio I\\resultado1\\model15.h5')

# Funções auxiliares (get_bert_embeddings, preprocess_input, predict) aqui...

@app.route('/', methods=['GET', 'POST'])
def index():
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

        # Fazer a previsão
        top_k_labels, top_k_probabilities, top_k_descriptions = predict(df_input, model1, categorical_features, numerical_features, text_columns, tokenizer, bert_model, scaler, encoder, code_to_description, top_k=5)

        # Preparar os resultados para exibição
        results = []
        for labels, probabilities, descriptions in zip(top_k_labels, top_k_probabilities, top_k_descriptions):
            result = [{"label": label, "probability": prob, "description": desc} for label, prob, desc in zip(labels, probabilities, descriptions)]
            results.append(result)

        return render_template('index.html', results=results)

    return render_template('index.html', results=None)

if __name__ == '__main__':
    app.run(debug=True)