import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Example text data and labels (1=positive, 0=negative)
texts = [
    "I love this house",
    "This view is amazing",
    "I hate this place",
    "I do not like this house",
    "I like this place"
]
labels = np.array([1, 1, 0, 0, 1])

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
print(f"Tokenized sequences: {sequences}")
maxlen = 5
X = pad_sequences(sequences, maxlen=maxlen)
print(f"Padded sequences: {X}")

# Build RNN model
model = keras.Sequential([
    layers.Embedding(input_dim=100, output_dim=8, input_length=maxlen),
    layers.SimpleRNN(8, activation='tanh'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, labels, epochs=100, verbose=0)

# Test prediction
test_text = ["I love this place"]
test_seq = tokenizer.texts_to_sequences(test_text)
print(f"Test sequence: {test_seq}")
test_pad = pad_sequences(test_seq, maxlen=maxlen)
print(f"test_pad: {test_pad}")
pred = model.predict(test_pad)
print(f"pred: {pred}")
print(f"Sentiment score: {pred[0][0]:.2f} (1=positive, 0=negative)")