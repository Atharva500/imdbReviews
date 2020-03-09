import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

imdb,info = tfds.load("imdb_reviews",with_info=True, as_supervised=True)

train_data,test_data = imdb['train'],imdb['test']
training_sentences=[]
training_labels=[]
test_sentences=[]
test_labels=[]

for s,l in train_data:
   training_sentences.append(str(s.numpy()))
   training_labels.append(l.numpy())
for s,l in test_data:   
   test_sentences.append(str(s.numpy()))
   test_labels.append(l.numpy())
training_labels_final = np.array(training_labels)
test_labels_final = np.array(test_labels)

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"


tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length,truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(test_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length,truncating=trunc_type)

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
# print(decode_review(padded[1]))
# print(training_sentences[1])


model = tf.keras.Sequential([
    keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    keras.layers.Flatten(),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 10
model.fit(padded,training_labels_final,epochs=num_epochs,validation_data=(testing_padded,test_labels_final))


e = model.layers[0]
weights = e.get_weights()[0]

import io
out_v = io.open('vecs.tsv' , 'w' , encoding='utf-8')
out_m = io.open('meta.tsv' , 'w' , encoding='utf-8')
for word_num in range(1,vocab_size):
    word=reverse_word_index[word_num]
    embeddings=weights[word_num]
    out_m.write(word+'\n')
    out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')
out_v.close()
out_m.close()


























