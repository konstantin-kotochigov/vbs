def model_lstm_1(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Attention(input_shape[1])(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def model_lstm_2(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=False))(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def model_lstm_3(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Attention(input_shape[1])(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def model_lstm_4(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Attention(input_shape[1])(x)
    x = Dense(64, activation="prelu")(x)
    x = Dense(32, activation="prelu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def model_lstm_5(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    x = Attention(input_shape[1])(x)
    x = Dense(64, activation="prelu")(x)
    x = Dense(32, activation="prelu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def model_lstm_6(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    x = Bidirectional(CuDNNLSTM(32, return_sequences=True))(inp)
    x = Bidirectional(CuDNNLSTM(16, return_sequences=True))(x)
    x = Attention(input_shape[1])(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def model_lstm_7(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    x = Bidirectional(CuDNNLSTM(256, return_sequences=True))(inp)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Attention(input_shape[1])(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def model_lstm_8(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    x = Bidirectional(CuDNNLSTM(8, return_sequences=True))(inp)
    x = Bidirectional(CuDNNLSTM(4, return_sequences=True))(x)
    x = Attention(input_shape[1])(x)
    x = Dense(4, activation="relu")(x)
    x = Dense(2, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def model_lstm_9(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    x = Bidirectional(CuDNNLSTM(8, return_sequences=True))(inp)
    x = Bidirectional(CuDNNLSTM(4, return_sequences=True))(x)
    x = Attention(input_shape[1])(x)
    x = Dense(4, activation="prelu")(x)
    x = Dense(2, activation="prelu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

models = [model_lstm_1(), model_lstm_2(), model_lstm_3(), model_lstm_4(), model_lstm_5(), model_lstm_6()]

result = []
for model in models:
    num_epochs = 25
    stats = custom_fit(X, y, num_epochs=num_epochs, num_splits=10)
    result.append((num_epochs, numpy.max(stats['mean'] - stats['std'])))
    pandas.DataFrame(result, columns=['param','quality']).to_csv(data_dir+"grid_search.csv", sep=";", index=False)

