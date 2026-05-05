import os
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def preprocess_sentence(s):
    s = s.lower().strip()
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = re.sub(r"[^a-zA-Z?.!,¿áéíóúÁÉÍÓÚñÑ]+", " ", s)
    return s.strip()

SAVE_PATH = './model/'

# Cargar modelo (formato nativo .keras no tiene el bug de score_mode)
model = tf.keras.models.load_model(os.path.join(SAVE_PATH, 'nmt_model.keras'))

# Cargar tokenizers
with open(os.path.join(SAVE_PATH, 'tokenizer_en.pkl'), 'rb') as f:
    en_tokenizer = pickle.load(f)
with open(os.path.join(SAVE_PATH, 'tokenizer_es.pkl'), 'rb') as f:
    es_tokenizer = pickle.load(f)

# Cargar configuracion
with open(os.path.join(SAVE_PATH, 'config.pkl'), 'rb') as f:
    config = pickle.load(f)
max_len_en = config['max_len_en']
max_len_es = config['max_len_es']


def translate(input_sentence):
    input_sentence = preprocess_sentence(input_sentence)
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
texto_a_traducir = "A carbon footprint is the amount of carbon dioxide pollution that we produce as a result of our activities. Some people try to reduce their carbon footprint because they are concerned about climate change."
resultado = translate(texto_a_traducir)
print(f"\nEnglish: {texto_a_traducir}")
print(f"Spanish: {resultado}")
