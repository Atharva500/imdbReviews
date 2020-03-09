import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

sentences = [
    'I am Atharva',
    'I am Ironman',
    'I dont think you are Ironman'
]

token = Tokenizer(num_words=100)
token.fit_on_texts(sentences)
word_index = token.word_index

#sequencing data line by line
seq = token.texts_to_sequences(sentences)

#padding
padding = pad_sequences(seq,padding='post')

print(padding)
# print(seq)
# print(word_index)