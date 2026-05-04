import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Attention, Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================================================================
# (You must keep the earlier part that defines max_len_en, max_len_es,
#  en_tokenizer and es_tokenizer – at least load them from files)
# ==============================================================================

SAVE_PATH = './model/'

# -------------------- 1. Minimal FIX for the Attention layer --------------------
class MyAttention(Layer):
    def __init__(self, use_scale=False, score_mode='dot', dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        if callable(score_mode):
            score_mode = 'dot'
        self.att = Attention(use_scale=use_scale, score_mode=score_mode, dropout=dropout)

    def call(self, inputs):
        return self.att(inputs)
# --------------------------------------------------------------------------------

# -------------------- 2. Load model, tokenizers & config ------------------------
model = tf.keras.models.load_model(
    os.path.join(SAVE_PATH, 'nmt_model.h5'),
    custom_objects={
        'Attention': MyAttention,
        'dot': K.dot
    }
)

with open(os.path.join(SAVE_PATH, 'tokenizer_en.pkl'), 'rb') as f:
    en_tokenizer = pickle.load(f)
with open(os.path.join(SAVE_PATH, 'tokenizer_es.pkl'), 'rb') as f:
    es_tokenizer = pickle.load(f)

with open(os.path.join(SAVE_PATH, 'config.pkl'), 'rb') as f:
    config = pickle.load(f)
max_len_en = config['max_len_en']
max_len_es = config['max_len_es']
# --------------------------------------------------------------------------------

# -------------------- 3. Inference function (unchanged) -------------------------
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


# -------------------- 4. HERE you put your sentence to translate ----------------
texto_a_traducir = "the house is big"          # <--- YOUR INPUT TEXT GOES HERE
resultado = translate(texto_a_traducir)
print(f"\nEnglish: {texto_a_traducir}")
print(f"Spanish: {resultado}")