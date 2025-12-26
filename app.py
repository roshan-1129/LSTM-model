
import streamlit as st

import numpy as np
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# function to predict next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences(
        [sequence],
        maxlen=max_sequence_len - 1,
        padding='pre'
    )
    predicted_probs = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""

# load model (IMPORTANT FIX HERE)
model = load_model(
    'lstm_text_generator.keras'
    
)

# load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# streamlit UI
st.header("Next Word Prediction using LSTM Model")
input_text = st.text_input("Enter your text:")

if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(
        model,
        tokenizer,
        input_text,
        max_sequence_len
    )
    st.write(f"Predicted Next Word: {next_word}")
