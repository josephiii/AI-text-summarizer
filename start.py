
from configs import hideTensorWarnings as htw
import tensorflow as tf
from keras import layers
import keras


# === Training Data ===
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

# === Tokenization ===
input_vector = layers.TextVectorization(max_tokens=100, output_mode='int', output_sequence_length=15)
output_vector = layers.TextVectorization(max_tokens=100, output_mode='int', output_sequence_length=15)
input_vector.adapt(texts)
output_vector.adapt(summaries)

input_tensor = input_vector(texts)
output_tensor = output_vector(summaries)

# === Hyperparameters ===
embedding_dim = 64
lstm_units = 64
vocab_size = 100

# === Encoder ===
encoder_input = keras.Input(shape=(15,))
encoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_input)
encoder_output, hidden_state, cell_state = layers.LSTM(lstm_units, return_state=True)(encoder_embedding)

# === Decoder ===
decoder_input = keras.Input(shape=(15,))
decoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_input)
decoder_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_output, decoder_h, decoder_c = decoder_lstm(decoder_embedding, initial_state=[hidden_state, cell_state])
decoder_output = layers.Dense(vocab_size, activation='softmax')(decoder_output)


# === Full Training Model ===
model = keras.Model([encoder_input, decoder_input], decoder_output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# === Shift summaries for teacher forcing ===
decoder_input_data = output_tensor[:, :-1]
decoder_target_data = output_tensor[:, 1:]
decoder_input_data = tf.pad(decoder_input_data, [[0, 0], [0, 1]])
decoder_target_data = tf.pad(decoder_target_data, [[0, 0], [0, 1]])

# === Train the Model ===
model.fit([input_tensor, decoder_input_data], decoder_target_data, epochs=100)

encoder_model = keras.Model(encoder_input, [hidden_state, cell_state])

decoder_embedding_layer = model.layers[3]
decoder_lstm_layer = model.layers[5]
decoder_dense_layer = model.layers[6]

def summarize(text):
    input_seq = input_vector([text])
    h, c = encoder_model.predict(input_seq)

    current_token = tf.constant([[0]]) 
    summary_ids = []

    for _ in range(15):
        x = decoder_embedding_layer(current_token)
        output, h, c = decoder_lstm_layer(x, initial_state=[h, c])
        logits = decoder_dense_layer(output)
        next_token_id = tf.argmax(logits[0, -1]).numpy()

        if next_token_id == 0:
            break

        summary_ids.append(next_token_id)
        current_token = tf.constant([[next_token_id]])

    vocab = output_vector.get_vocabulary()
    return ' '.join([vocab[i] for i in summary_ids if i > 0])


print(summarize("It is very cold outside I am going to put on a jacket"))

