from flask import Flask, request, render_template,jsonify
import torch
import joblib
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from tensorflow.keras.models import load_model # type: ignore

app = Flask(__name__)

# Carregando o modelo e outros componentes necessários
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Carregar scaler e encoder treinados
scaler = joblib.load('https://github.com/danielsilva83/app_assistAI_parts/blob/main/static/scaler17.pkl')
encoder = joblib.load('https://github.com/danielsilva83/app_assistAI_parts/blob/main/static/label_encoder17.pkl')

# Carregar o CSV com o mapeamento de 'codigo_solicitado' para suas descrições
csv_path = 'C:\\Users\\daniel.silva\\Documents\\cits\\app_assistAI_parts\\static\\cod_solic3.csv'
code_df = pd.read_csv(csv_path)
code_to_description = dict(zip(code_df['codigo_solicitado'], code_df['cod_solic_desc']))

# Carregar o arquivo de exemplos
examples_csv_path = 'https://github.com/danielsilva83/app_assistAI_parts/blob/main/static/amostra_exemplo.csv'
examples_df = pd.read_csv(examples_csv_path)

# Carregar o modelo
model1 = load_model('https://github.com/danielsilva83/app_assistAI_parts/blob/main/static/model17.h5')
#def load_examples(start=0, count=50):
#    print( examples_df.iloc[start:start + count].to_dict(orient='records'))
#    return examples_df.iloc[start:start + count].to_dict(orient='records')
def load_examples(n=7):
    df = pd.read_csv(examples_csv_path)
    return df.sample(n).to_dict(orient='records')
# Função para extrair embeddings dos textos
def get_bert_embeddings(texts, tokenizer, bert_model, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        embeddings.append(batch_embeddings)
        del inputs, outputs
        torch.cuda.empty_cache()
    return np.vstack(embeddings)

def preprocess_input(df, numerical_features, text_columns, tokenizer, bert_model, scaler):
    # Remover NaN em 'codigo_solicitado'
    df.fillna('', inplace=True)
    df.drop_duplicates(inplace=True)

    # Processamento dos textos com BERT
    # Aplica a função nos campos textuais em batches
    text_embeddings = []
    for column in text_columns:
        embeddings = get_bert_embeddings(df[column].tolist(), tokenizer, bert_model, batch_size=32)
        text_embeddings.append(embeddings)
    
    # Concatenar os embeddings textuais
    all_text_embeddings = np.hstack(text_embeddings)
    # Verificação das dimensões
    if df[numerical_features].shape[0] == all_text_embeddings.shape[0]:
        X_final = np.hstack([df[numerical_features], all_text_embeddings])
    else:
        raise ValueError("Dimensões incompatíveis entre X_normalized e text embeddings.")
    
    # Normalização dos dados numéricos
    X_final = scaler.transform(X_final)
   
    return X_final

def predict(df, model, numerical_features, text_columns, tokenizer, bert_model, scaler, encoder, code_to_description, top_k=5):
    X_processed = preprocess_input(df, numerical_features, text_columns, tokenizer, bert_model, scaler)
    predictions = model.predict(X_processed)
    top_k_indices = np.argsort(predictions, axis=1)[:, -top_k:][:, ::-1]  # Pega os top_k índices, ordenados por probabilidade
    top_k_labels = [encoder.inverse_transform(indices) for indices in top_k_indices]
    top_k_probabilities = np.sort(predictions, axis=1)[:, -top_k:][:, ::-1]  # Pega as top_k probabilidades, ordenadas
    top_k_descriptions = [[code_to_description[label] for label in labels] for labels in top_k_labels]
    return top_k_labels, top_k_probabilities, top_k_descriptions

@app.route('/', methods=['GET', 'POST'])
def index():
    examples = load_examples()
    
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