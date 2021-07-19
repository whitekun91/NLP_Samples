from tensorflow.keras import preprocessing
import numpy as np
import text_data

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(text_data.samples)

sequences = tokenizer.texts_to_sequences(text_data.samples)
input_sequences = np.array(sequences)

labels = np.array(text_data.label)

word_index = tokenizer.word_index
