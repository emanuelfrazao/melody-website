import numpy as np
import tensorflow
from keras import Model, Input, layers

def init_lstm_bidirect(hidden_dim, ohe_embed_dim):
    # 1. Define encoder
    e_num_input = Input(shape=(None, 5), name='input_num_feat')
    e_num = layers.Masking()(e_num_input)
    
    e_ohe_input = Input(shape=(None, 13), name='input_cat_feat')
    e_ohe = layers.Masking()(e_ohe_input)
    e_ohe = layers.Dense(ohe_embed_dim, use_bias=False)(e_ohe)

    e_out = layers.concatenate([e_num, e_ohe])
    e_out, e_h1, e_m1, e_h2, e_m2 = layers.Bidirectional(
        layers.LSTM(hidden_dim, return_state=True, name='encoder1'),
        merge_mode='sum'
    )(e_out)

    e_hidden = layers.concatenate([e_h1, e_h2])
    e_memory = layers.concatenate([e_m1, e_m2])
    e_state = [e_hidden, e_memory]

    encoder = Model([e_num_input, e_ohe_input], e_state)

    # 2. Define decoder
    d_num_input = Input(shape=(None, 4), name='input_num_target')
    d_num = layers.Masking()(d_num_input)

    d_ohe_input = Input(shape=(None, 13), name='input_cat_target')
    d_ohe = layers.Masking()(d_ohe_input)
    d_ohe = layers.Dense(ohe_embed_dim, use_bias=False)(d_ohe)

    d_out = layers.concatenate([d_num, d_ohe])
    d_out, *d_state = layers.LSTM(2*hidden_dim, return_sequences=True, return_state=True)(d_out, initial_state=e_state)

    d_num_output = layers.Dropout(0.1)(d_out)
    d_num_output = layers.Dense(4, activation='linear', name='output_num')(d_num_output)

    d_ohe_output = layers.Dropout(0.1)(d_out)
    d_ohe_output = layers.Dense(13, activation='softmax', name='output_cat')(d_ohe_output)

    decoder = Model([d_num_input, d_ohe_input, *e_state], [d_num_output, d_ohe_output, *d_state])

    # 3. Define encoder-decoder
    model = Model([e_num_input, e_ohe_input, d_num_input, d_ohe_input], [d_num_output, d_ohe_output])

    return model, encoder, decoder


_, encoder, decoder = init_lstm_bidirect(64, 7)
encoder.load_weights('weights/encoder/')
decoder.load_weights('weights/decoder/')

def generate(rhythm_num, rhythm_ohe, temperature=0.5, max_size=125):
    # Get embedding from encoder
    rhythm_num = np.expand_dims(rhythm_num, axis=0)
    rhythm_ohe = np.expand_dims(rhythm_ohe, axis=0)
    embedding = encoder([rhythm_num, rhythm_ohe])

    # Set initial melody note
    last_note_num = np.zeros((1, 1, 4)) # (1, 1, 4) make it one sequence with one obs
    last_note_ohe = np.zeros((1, 1, 13)) # (1, 1, 13) make it one sequence with one obs

    notes = []

    # Generate new notes and keep feeding them as input the model
    for i in range(max_size):
        # Run through decoder
        last_note_num, last_note_softmax, *embedding = decoder([last_note_num, last_note_ohe, *embedding])
        last_note_softmax = last_note_softmax[0, 0].numpy().astype('float64')

        # Sample from softmax
        temp_softmax = np.log(last_note_softmax) / temperature  # unroll softmax to apply temperature
        exp_softmax = np.exp(temp_softmax)                      # intermediate step
        last_note_softmax = exp_softmax / np.sum(exp_softmax)   # get softmax with temperature

        last_note_ohe = np.random.multinomial(1, last_note_softmax, size=1)
        last_note = np.argmax(last_note_ohe)

        # Grab last note and append it to lead sequence (shifted)
        last_note_ohe = np.expand_dims(last_note_ohe, axis=0)
        notes.append([*last_note_num[0][0].numpy(), last_note])
    
    return np.array(notes)