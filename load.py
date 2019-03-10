import pandas
import pandas as pd
import pyarrow.parquet as pq # Used to read the data
import numpy as np
import numpy
from tqdm import tqdm
from keras.models import Sequential, Model
from keras.layers import *
from keras import optimizers
from keras import backend as K
from keras.callbacks import *
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from keras.engine.topology import Layer
import datetime
import tensorflow as tf

# cd /srv/vbalaev/vsp_power/

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

def matthews_correlation(y_true, y_pred):
    mcc = []
    for threshold in range(0, 100, 5):
        y_pred_pos = numpy.where(y_pred>threshold/100, 1, 0)
        y_pred_neg = 1 - y_pred_pos
        y_pos = y_true
        y_neg = 1 - y_pos
        tp = numpy.sum(y_true * y_pred_pos)
        tn = numpy.sum(y_neg * y_pred_neg)
        fp = numpy.sum(y_neg * y_pred_pos)
        fn = numpy.sum(y_pos * y_pred_neg)
        numerator = (tp * tn - fp * fn)
        denominator = numpy.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc.append(round(numerator / (denominator + 0.001), 5))
    return mcc


data_dir = "/home/ubuntu/"
temp_dir = "/home/ubuntu/vbs_temp/"

df_train = pandas.read_csv(data_dir + 'metadata_train.csv')
df_train = df_train.set_index(['id_measurement', 'phase'])
df_train.head()

def min_max_transf(ts):
     ts_std = (ts + abs(-128)) / (127 + abs(-128))
     return (ts_std * (2) - 1)

sample_size = 800000
def transform_ts(ts, chunk_size):
    ts_std = min_max_transf(ts)
    bucket_size = chunk_size
    new_ts = []
    for i in range(0, sample_size, bucket_size):
        ts_range = ts_std[i:i + bucket_size]
        mean = ts_range.mean()
        std = ts_range.std() # standard deviation
        std_top = mean + std # I have to test it more, but is is like a band
        std_bot = mean - std
        percentil_calc = np.percentile(ts_range, [0, 1, 25, 50, 75, 99, 100])
        max_range = percentil_calc[-1] - percentil_calc[0] # this is the amplitude of the chunk
        relative_percentile = percentil_calc - mean # maybe it could heap to understand the asymmetry
        new_ts.append(np.concatenate([np.asarray([mean, std, std_top, std_bot, max_range]),percentil_calc, relative_percentile]))
    return np.asarray(new_ts)

def prep_data(start, end, chunk_size):
    praq_train = pq.read_pandas('train.parquet', columns=[str(i) for i in range(start, end)]).to_pandas()
    X = []
    y = []
    for id_measurement in tqdm(df_train.index.levels[0].unique()[int(start/3):int(end/3)]):
        X_signal = []
        for phase in [0,1,2]:
            signal_id, target = df_train.loc[id_measurement].loc[phase]
            if phase == 0:
                y.append(target)
            X_signal.append(transform_ts(praq_train[str(signal_id)], chunk_size))
        X_signal = np.concatenate(X_signal, axis=1)
        X.append(X_signal)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y

# this code is very simple, divide the total size of the df_train into two sets and process it
def load_all(chunk_size=5000):
    X = []
    y = []
    total_size = len(df_train)
    for ini, end in [(0, int(total_size/2)), (int(total_size/2), total_size)]:
        X_temp, y_temp = prep_data(ini, end, chunk_size)
        X.append(X_temp)
        y.append(y_temp)
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X,y





class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True
    def compute_mask(self, input, input_mask=None):
        return None
    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)
    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

def model_lstm(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=False))(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def custom_fit(X, y, num_epochs, num_splits, batch_size=256):
    # splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=2019).split(X_transformed, y))
    splits = list(StratifiedShuffleSplit(n_splits=num_splits, test_size=0.25).split(X, y))
    quality_list = []
    for idx, (train_idx, val_idx) in enumerate(splits):
        K.clear_session() # I dont know what it do, but I imagine that it "clear session" :)
        print("Beginning fold {}".format(idx+1))
        train_X, train_y, val_X, val_y = X[train_idx], y[train_idx], X[val_idx], y[val_idx]
        model = model_lstm(train_X.shape)
        # ckpt = ModelCheckpoint('/srv/kkotochigov/weights_{}.h5'.format(idx), save_best_only=True, save_weights_only=True, verbose=1, monitor='val_matthews_correlation', mode='max')
        model.fit(train_X, train_y, batch_size=batch_size, epochs=num_epochs, validation_data=[val_X, val_y])
        # model.load_weights('/srv/kkotochigov/weights_{}.h5'.format(idx))
        y_pred = model.predict(val_X, batch_size=batch_size)
        quality = matthews_correlation(val_y, y_pred.reshape(y_pred.shape[0],))
        quality_list.append(quality)
    stats = numpy.stack(quality_list)
    best_threshold = numpy.argmax(numpy.mean(stats, axis=0))
    return {"threshold":5*best_threshold, "mean":numpy.mean(stats, axis=0), "std":numpy.std(stats, axis=0)}

for chunk_size in [1000,3000,5000]:
    X, y = load_all(chunk_size)
    numpy.save(temp_dir + "X_train_{}".format(chunk_size), X)
    numpy.save(temp_dir + "y_train_{}".format(chunk_size), y)
    # X_test = reduce_dimensionality(X_test_raw, chunk_size)
    # numpy.save(temp_dir + "X_test_{}".format(chunk_size), X_test)



chunk_size = 5000
X = numpy.load(temp_dir+"X_train_{}.npy".format(chunk_size))
y = numpy.load(temp_dir+"y_train_{}.npy".format(chunk_size))
# X_test = numpy.load(temp_dir+"X_test_{}.npy".format(chunk_size))
# y = numpy.asarray(y_raw)
# stats = custom_fit(X, y)

result = []
for num_epochs in [50]:
    stats = custom_fit(X, y, num_epochs=num_epochs, num_splits=25)
    result.append((chunk_size, numpy.max(stats['mean'] - stats['std']), 5 * numpy.argmax(stats['mean'] - stats['std']),numpy.max(stats['mean']), 5 * numpy.argmax(stats['mean'])))
    pandas.DataFrame(result, columns=['param', 'lcb_quality', 'lcb_threshold', 'quality', 'threshold']).to_csv(data_dir + "grid_search.csv", sep=";", index=False)

result = []
num_epochs = 50
for chunk_size in [1000, 10000]:
    print("chunk_size={}".format(chunk_size))
    X = numpy.load(temp_dir + "X_train_{}.npy".format(chunk_size))
    X_test = numpy.load(temp_dir + "X_test_{}.npy".format(chunk_size))
    y = numpy.asarray(y_raw)
    stats = custom_fit(X, y, num_epochs=num_epochs, num_splits=10, batch_size=128)
    result.append((chunk_size, numpy.max(stats['mean'] - stats['std']), 5*numpy.argmax(stats['mean'] - stats['std']), numpy.max(stats['mean']), 5*numpy.argmax(stats['mean'])))
    pandas.DataFrame(result, columns=['param','lcb_quality','lcb_threshold', 'quality','threshold']).to_csv(data_dir+"grid_search.csv", sep=";", index=False)

result = []
for model in models:
    stats = custom_fit(X, y, num_epochs=num_epochs, num_splits=10)
    result.append((num_epochs, numpy.max(stats['mean'] - stats['std'])))
    pandas.DataFrame(result, columns=['param','quality']).to_csv(data_dir+"grid_search.csv", sep=";", index=False)

model = model_lstm(X.shape)
optimal_epochs = 100
model.fit(X, y, batch_size=1024, epochs=optimal_epochs)
y_test = model.predict(X_test)

optimal_threshold = 0.4
y_test = y_test.reshape(len(y_test))
y_test = numpy.concatenate([[x,x,x] for x in y_test])
filename = data_dir+"vbs/submission_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
pandas.DataFrame({"signal_id": range(8712, 29049), "target":numpy.where(y_test > optimal_threshold, 1, 0)}).to_csv(filename, index=False)

# Submit solution
# submission_text = "set 50 epochs"
# os.system("kaggle competitions submit -f {} -m '{}' ".format(filename, submission_text))