#!/usr/bin/env python3
# ==============================================================================
# Neural Machine Translation ENG to SPA
# Arquitectura: Encoder-Decoder + Atención + Pre-trained Embeddings (SpaCy)
# ==============================================================================

# ==============================================================================
# EJERCICIO ACADÉMICO: MACHINE TRANSLATION (EN -> ES)
# Arquitectura: Encoder-Decoder + Atención + Pre-trained Embeddings (SpaCy)
# ==============================================================================

# 1. IMPORTACIÓN DE LIBRERÍAS
import os
import re
import subprocess
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Embedding, Attention, Concatenate, TimeDistributed,Dropout
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
from tensorflow.keras.callbacks import ReduceLROnPlateau


# ==============================================================================
# Dataset ENG-SPA
# ==============================================================================

#subprocess.run(["wget", "-q", "http://www.manythings.org/anki/spa-eng.zip"], check=True)
#subprocess.run(["unzip", "-o", "spa-eng.zip"], check=True)


def preprocess_sentence(s):
    s = s.lower().strip()
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = re.sub(r"[^a-zA-Z?.!,¿áéíóúÁÉÍÓÚñÑ]+", " ", s)
    return s.strip()


def load_data(path, num_samples=144000):
    en_sentences = []
    es_sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    for line in lines[:num_samples]:
        parts = line.split('\t') #Split the different columns
        if len(parts) >= 2:
            en_sentences.append(preprocess_sentence(parts[0])) 
            es_sentences.append('<start> ' + preprocess_sentence(parts[1]) + ' <end>')

    # Mix randomly the sentences in the datasets
    indices = np.random.permutation(len(en_sentences))
    en_sentences = [en_sentences[i] for i in indices]
    es_sentences = [es_sentences[i] for i in indices]

    return en_sentences, es_sentences



english_texts, spanish_texts = load_data('spa.txt', num_samples=144000)

print(f"Muestra del dataset: {english_texts[100]} -> {spanish_texts[100]}")

# Configuración de rutas
SAVE_PATH = './model/'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)



# ==============================================================================
# 2. CARGA DE DATOS Y PREPROCESAMIENTO
# ==============================================================================


# Tokenización
en_tokenizer = Tokenizer(filters='')
en_tokenizer.fit_on_texts(english_texts)
en_seq = en_tokenizer.texts_to_sequences(english_texts)

es_tokenizer = Tokenizer(filters='')
es_tokenizer.fit_on_texts(spanish_texts)
es_seq = es_tokenizer.texts_to_sequences(spanish_texts)

# Calculamos las longitudes máximas reales del dataset cargado
max_len_en = max(len(s) for s in en_seq)
max_len_es = max(len(s) for s in es_seq)

encoder_input_data = pad_sequences(en_seq, maxlen=max_len_en, padding='post')
decoder_input_data = pad_sequences(es_seq, maxlen=max_len_es, padding='post')

# Crear el target desplazado
decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:, :-1] = decoder_input_data[:, 1:]

# Vocabularios
en_vocab_size = len(en_tokenizer.word_index) + 1
es_vocab_size = len(es_tokenizer.word_index) + 1

# ==============================================================================
# 3. EMBEDDINGS PREENTRENADOS CON SPACY
# ==============================================================================

subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_md"], check=True)
subprocess.run([sys.executable, "-m", "spacy", "download", "es_core_news_md"], check=True)

nlp_en = spacy.load("en_core_web_md")
nlp_es = spacy.load("es_core_news_md")


def get_embedding_matrix(tokenizer, nlp_model, vocab_size, dim=300):
    matrix = np.zeros((vocab_size, dim))
    for word, i in tokenizer.word_index.items():
        if i < vocab_size:
            matrix[i] = nlp_model(word).vector
    return matrix


en_embedding_matrix = get_embedding_matrix(en_tokenizer, nlp_en, en_vocab_size)
es_embedding_matrix = get_embedding_matrix(es_tokenizer, nlp_es, es_vocab_size)

# Inicializar tokens especiales con vectores aleatorios (SpaCy les da ceros)
for token in ['<start>', '<end>']:
    if token in es_tokenizer.word_index:
        idx = es_tokenizer.word_index[token]
        es_embedding_matrix[idx] = np.random.uniform(-0.05, 0.05, 300)

# ==============================================================================
# 4. DEFINICIÓN DEL MODELO (ENCODER - DECODER + ATTENTION)
# ==============================================================================
latent_dim = 300  # Dimensión de las unidades LSTM

# --- ENCODER ---
encoder_inputs = Input(shape=(max_len_en,), name="Enc_Input")
enc_emb = Embedding(
    en_vocab_size, 300, weights=[en_embedding_matrix], mask_zero=True, trainable=True
)(encoder_inputs)
enc_emb = Dropout(0.5)(enc_emb)
encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="Enc_LSTM")
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# --- DECODER ---
decoder_inputs = Input(shape=(max_len_es,), name="Dec_Input")
dec_emb_layer = Embedding(
    es_vocab_size, 300, weights=[es_embedding_matrix], mask_zero=True, trainable=True
)

dec_emb = dec_emb_layer(decoder_inputs)
dec_emb = Dropout(0.5)(dec_emb)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,  name="Dec_LSTM")
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

# --- MECANISMO DE ATENCIÓN (Luong Style) ---
attention_layer = Attention(name="Attention_Layer")
attn_out = attention_layer([decoder_outputs, encoder_outputs])

# Concatenamos el contexto de atención con la salida del decoder
decoder_concat_input = Concatenate(axis=-1, name="Concat_Layer")(
    [decoder_outputs, attn_out]
)

# Capa densa final
decoder_concat_input = Dropout(0.5)(decoder_concat_input)
decoder_dense = TimeDistributed(Dense(es_vocab_size, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# Definir modelo de entrenamiento

callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',                                                                                                                                                                              
    factor=0.5,        # reduce lr a la mitad
    patience=2,        # espera 2 epochs sin mejora                                                                                                                                                  
    min_lr=1e-6        # no bajar de esto
)   


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ============================e==================================================
# 5. ENTRENAMIENTO Y PERSISTENCIA
# ==============================================================================

print("\nIniciando entrenamiento...")
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=256,
    epochs=100,
    validation_split=0.1,
    callbacks=[callback,reduce_lr],
)

# Guardar el modelo completo
model.save(os.path.join(SAVE_PATH, 'nmt_model.keras'))
print(f"Modelo guardado en: {SAVE_PATH}")

# Guardar tokenizers
with open(os.path.join(SAVE_PATH, 'tokenizer_en.pkl'), 'wb') as f:
    pickle.dump(en_tokenizer, f)

with open(os.path.join(SAVE_PATH, 'tokenizer_es.pkl'), 'wb') as f:
    pickle.dump(es_tokenizer, f)

# Guardar configuración necesaria para inferencia
with open(os.path.join(SAVE_PATH, 'config.pkl'), 'wb') as f:
    pickle.dump({'max_len_en': max_len_en, 'max_len_es': max_len_es}, f)

print("Entrenamiento completado. Modelo y artefactos guardados en:", SAVE_PATH)
