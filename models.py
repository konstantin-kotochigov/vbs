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
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def model_lstm_5(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    x = Attention(input_shape[1])(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
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
    x = Dense(4, activation="relu")(x)
    x = Dense(2, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# for chunk_size in [10000, 1000]:
#     X, y = load_all(chunk_size)
#     numpy.save(temp_dir + "X_train_{}".format(chunk_size), X)
#     numpy.save(temp_dir + "y_train_{}".format(chunk_size), y)


chunk_size = 5000
inp_shape = X.shape
models = [model_lstm_1(inp_shape), model_lstm_2(inp_shape), model_lstm_3(inp_shape), model_lstm_4(inp_shape), model_lstm_5(inp_shape), model_lstm_6(inp_shape), model_lstm_7(inp_shape), model_lstm_8(inp_shape), model_lstm_9(inp_shape)]

result = []
for model_num, model in enumerate(models):
    num_epochs = 50
    stats = custom_fit(X, y, num_epochs=num_epochs, num_splits=20)
    model_result = ("model{}".format(model_num), numpy.max(stats['mean'] - stats['std']), 5 * numpy.argmax(stats['mean'] - stats['std']), numpy.max(stats['mean']), 5 * numpy.argmax(stats['mean']))
    print(model_result)
    result.append(model_result)
    pandas.DataFrame(result, columns=['param', 'lcb_quality', 'lcb_threshold', 'quality', 'threshold']).to_csv(data_dir + "grid_search.csv", sep=";", index=False)

