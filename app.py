from flask import Flask, request, render_template,jsonify
import torch
import pandas as pd
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from tensorflow.keras.models import load_model # type: ignore
from src.load_files_bert import load_files_bert
from src.load_examples import load_examples, load_examples_ini
from src.predict import predict
from src.process_input import get_bert_embeddings, preprocess_input


app = Flask(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Carregar o CSV com o mapeamento de 'codigo_solicitado' para suas descrições
csv_path = './static/cod_solic3.csv'
code_df = pd.read_csv(csv_path)
code_to_description = dict(zip(code_df['codigo_solicitado'], code_df['cod_solic_desc']))


# Carregar o arquivo de exemplos
examples_csv_path = './static/amostra_exemplo.csv'
examples_df = pd.read_csv(examples_csv_path)

model1, encoder, scaler = load_files_bert()

load_examples_ini(examples_df, start=150, count=10)
load_examples(examples_csv_path, n=7)


@app.route('/', methods=['GET', 'POST'])
def index():
    examples = load_examples_ini()

    if request.method == 'POST':
        modelo = request.form['modelo']
        problema = request.form['problema']
        tipo_equipamento = request.form['tipo_equipamento']

        # Criar o DataFrame com os dados de entrada
        data = {
            'modelo': [modelo],
            'problema': [problema],
            'tipo_equipamento': [tipo_equipamento]
        }
        df_input = pd.DataFrame(data)

        numerical_features = ['modelo']

        text_columns = ['problema', 'tipo_equipamento']
        df_input = preprocess_input()
        predict(df_input, model, numerical_features, text_columns, tokenizer, bert_model, scaler, encoder, code_to_description, top_k=5)

        get_bert_embeddings(df_input, numerical_features, text_columns, tokenizer, bert_model, scaler) 

        # Fazer a previsão
        top_k_labels, top_k_probabilities, top_k_descriptions = predict(df_input, model1, numerical_features, text_columns, tokenizer, bert_model, scaler, encoder, code_to_description, top_k=5)

        # Preparar os resultados para exibição
        results = []
        for labels, probabilities, descriptions in zip(top_k_labels, top_k_probabilities, top_k_descriptions):
            result = [{"label": label, "probability": prob, "description": desc} for label, prob, desc in zip(labels, probabilities, descriptions)]
            results.append(result)

        return render_template('index.html', results=results, examples=examples)

    return render_template('index.html', results=None, examples=examples)

@app.route('/more_examples', methods=['GET'])
def more_examples():
    examples = load_examples(7)
    return jsonify(examples)

if __name__ == '__main__':
    app.run(debug=True)