# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 11:38:58 2026

@author: SAMRUDDHI
"""
# A. Data Preparation 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import skipgrams 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, Lambda, Dense 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
# Sample text corpus 
corpus = [ 
"deep learning is very interesting", 
"natural language processing uses deep learning", 
"word embeddings represent words as vectors", 
"cbow predicts a target word from context words" 
] 
# Tokenize the text 
tokenizer = Tokenizer() 
tokenizer.fit_on_texts(corpus) 
word_index = tokenizer.word_index 
vocab_size = len(word_index) + 1 
print("Vocabulary:") 
print(word_index) 
print("\nVocabulary Size:", vocab_size) 
# Convert sentences into sequences of integers 
sequences = tokenizer.texts_to_sequences(corpus) 
print("\nTokenized Sequences:") 
print(sequences) 
# B. Generate CBOW Training Data 
window_size = 2 
contexts = [] 
targets = [] 
for sequence in sequences:
    for i in range(window_size, len(sequence) - window_size):
        context = (
            sequence[i - window_size:i]
            + sequence[i + 1:i + window_size + 1]
        )

        target = sequence[i]

        contexts.append(context)
        targets.append(target)
contexts = np.array(contexts, dtype="int32") 
targets = np.array(targets, dtype="int32") 
print("\nContext Words:") 
print(contexts) 
print("\nTarget Words:") 
print(targets) 
# Convert target words to one-hot vectors 
targets_one_hot = tf.keras.utils.to_categorical( 
targets, num_classes=vocab_size 
) 
# C. Build and Train the CBOW Model 
embedding_dim = 50 
model = Sequential([ 
Embedding( 
input_dim=vocab_size, 
output_dim=embedding_dim, 
input_length=window_size * 2 
), 
Lambda(lambda x: tf.reduce_mean(x, axis=1)), 
Dense(vocab_size, activation="softmax") 
]) 
model.compile( 
optimizer="adam", 
loss="categorical_crossentropy", 
metrics=["accuracy"] 
) 
model.summary() 
print("\nTraining the CBOW Model...") 
model.fit( 
contexts, 
targets_one_hot, 
epochs=100, 
verbose=0 
) 
print("Training Completed!") 
# D. Output: Predict the Target Word 
index_word = {index: word for word, index in word_index.items()} 
sample_context = contexts[0:1] 
predictions = model.predict(sample_context, verbose=0) 
predicted_index = np.argmax(predictions[0]) 
actual_index = targets[0] 
print("\n--- CBOW Output ---") 
print("Context Words:", 
[index_word[i] for i in sample_context[0]]) 
print("Actual Target Word:", index_word[actual_index]) 
print("Predicted Target Word:", index_word[predicted_index]) 
# Display word embeddings 
embedding_weights = model.layers[0].get_weights()[0] 
print("\nWord Embedding for 'learning':") 
print(embedding_weights[word_index["learning"]]) 
print("\nCBOW Practical Executed Successfully!")