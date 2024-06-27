from src.preprocess_spacy import *

def predict(df, model, text_columns, scaler, encoder, code_to_description, top_k=5):
    X_processed = preprocess_input(df, text_columns, scaler)
    predictions = model.predict(X_processed)
    top_k_indices = np.argsort(predictions, axis=1)[:, -top_k:][:, ::-1]
    top_k_probabilities = np.sort(predictions, axis=1)[:, -top_k:][:, ::-1] * 100 
    top_k_labels = [encoder.inverse_transform(indices) for indices in top_k_indices]
    top_k_descriptions = [[code_to_description[label] for label in labels] for labels in top_k_labels]
    return top_k_labels, top_k_probabilities, top_k_descriptions