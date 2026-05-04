# 1. IMPORTACIÓN DE LIBRERÍAS
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy




spanish_texts = ['<start> ' + s + ' <end>' for s in spanish_texts]

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
# Inferencia
# ==============================================================================

# Cargar el modelo (simula sesión nueva)
model = load_model(os.path.join(SAVE_PATH, 'nmt_model.h5'))

with open(os.path.join(SAVE_PATH, 'tokenizer_en.pkl'), 'rb') as f:
    en_tokenizer = pickle.load(f)

# ==============================================================================
# 6. INFERENCIA (TRADUCCIÓN DE NUEVOS TEXTOS)
# ==============================================================================


def translate(input_sentence):
    input_seq = en_tokenizer.texts_to_sequences([input_sentence])
    input_seq = pad_sequences(input_seq, maxlen=max_len_en, padding='post')

    decoded_sentence = '<start>'

    for i in range(max_len_es):
        target_seq = es_tokenizer.texts_to_sequences([decoded_sentence])
        target_seq = pad_sequences(target_seq, maxlen=max_len_es, padding='post')

        output_tokens = model.predict([input_seq, target_seq], verbose=0)

        sampled_token_index = np.argmax(output_tokens[0, i, :])
        sampled_char = es_tokenizer.index_word.get(sampled_token_index, '')

        if sampled_char == '<end>' or sampled_char == '':
            break

        decoded_sentence += ' ' + sampled_char

    return decoded_sentence.replace('<start>', '').strip()


# --- PRUEBA ---
texto_a_traducir = "the house is big"
resultado = translate(texto_a_traducir)
print(f"\nEnglish: {texto_a_traducir}")
print(f"Spanish: {resultado}")
