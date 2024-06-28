import pandas as pd

def load_examples(examples_csv_path, n=7):
    df = pd.read_csv(examples_csv_path)
    return df.sample(n).to_dict(orient='records')
# Função para extrair embeddings dos textos

def load_examples_ini(examples_df, start=150, count=10):
    print( examples_df.iloc[start:start + count].to_dict(orient='records'))
    return examples_df.iloc[start:start + count].to_dict(orient='records')