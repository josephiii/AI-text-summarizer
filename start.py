
from configs import hideTensorWarnings as htw
import tensorflow as tf
from keras import layers
import keras
import pandas as pd
import numpy as np

#training info
texts = [
    'It is very cold outside I am going to put on a jacket',
    'The sky is dark and stormy I think it will rain',
    'I am watching a show about wizards'
]

summaries = [
    'cold outside put on jacket', 
    'sky dark stormy it will rain',
    'watching show about wizards'
]

#tokenization and padding
input_vector = layers.TextVectorization(
    max_tokens = 100,
    output_mode = 'int',
    output_sequence_length = 15
)
input_vector.adapt(texts)


output_vector = layers.TextVectorization(
    max_tokens = 100,
    output_mode = 'int', 
    output_sequence_length = 15
)
output_vector.adapt(summaries)

input_tensor = input_vector(texts)
output_tensor = output_vector(summaries)

embedding_dim = 64 #this is the number of float values in our vectors
lstm_units = 64 #this is the number of words our lstm stores in memory
vocab_size = 100 #how many total words our model can store without [UNK] (top 100 most important)

#encoder
encoder_input = keras.Input(shape=(15,))
encoder_embedding =layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_input)
encoder_output, hidden_state, cell_state = layers.LSTM(lstm_units, return_state=True,)(encoder_embedding) 

#decoder




