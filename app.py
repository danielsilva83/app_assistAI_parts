from flask import Flask, request, render_template,jsonify
import pandas as pd
import numpy as np
from src.load_file_spacy import load_files_spacy
from src.predict_spacy import predict


app = Flask(__name__)

# Carregar o CSV com o mapeamento de 'codigo_solicitado' para suas descrições
csv_path = './static/cod_solic3.csv'
code_df = pd.read_csv(csv_path)
code_to_description = dict(zip(code_df['codigo_solicitado'], code_df['cod_solic_desc']))

# Carregar o arquivo de exemplos
examples_csv_path = './static/amostra_exemplo.csv'
examples_df = pd.read_csv(examples_csv_path)

def load_examples_ini(start=150, count=10):
    print( examples_df.iloc[start:start + count].to_dict(orient='records'))
    return examples_df.iloc[start:start + count].to_dict(orient='records')

def load_examples(n=7):
    df = pd.read_csv(examples_csv_path)
    return df.sample(n).to_dict(orient='records')
    
scaler, encoder, model = load_files_spacy() 

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
      
        text_columns = ['problema', 'tipo_equipamento']
        
        # Fazer a previsão
        top_k_labels, top_k_probabilities, top_k_descriptions = predict(df_input, model, text_columns, scaler, encoder, code_to_description, top_k=5)

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