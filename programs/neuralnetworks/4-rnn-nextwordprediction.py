import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example corpus
texts = [
    "I love this house",
    "This view is amazing",
    "I hate this place",
    "I do not like this house",
    "I like this place"
]

# Prepare sequences for next-word prediction
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
labels = []

for line in texts:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i])
        labels.append(token_list[i])

# Pad sequences
max_seq_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')
labels = np.array(labels)

# Build the model
model = keras.Sequential([
    layers.Embedding(total_words, 10, input_length=max_seq_len),
    layers.SimpleRNN(32),
    layers.Dense(total_words, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(input_sequences, labels, epochs=500, verbose=0)

# Predict the next word
seed_text = "I love"
token_list = tokenizer.texts_to_sequences([seed_text])[0]
token_list = pad_sequences([token_list], maxlen=max_seq_len, padding='pre')
predicted_probs = model.predict(token_list, verbose=0)
predicted_index = np.argmax(predicted_probs)
predicted_word = ""
for word, index in tokenizer.word_index.items():
    if index == predicted_index:
        predicted_word = word
        break

print(f"Input: '{seed_text}'")
print(f"Predicted next word: '{predicted_word}'")