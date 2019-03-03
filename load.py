import pandas
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

# cd /srv/vbalaev/vsp_power/

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
temp_dir = "/srv/kktoochigov/vbs/"

train = pandas.read_parquet("train.parquet")
test = pandas.read_parquet("test.parquet")
meta = pandas.read_csv("metadata_train.csv")
meta_test = pandas.read_csv("metadata_test.csv")

n = int(train.shape[1]/3) # 2904
N = train.shape[0] # 800000

n_test = int(test.shape[1]/3) # 6679
N_test = test.shape[0] # 800000

X_raw, y_raw = [], []
X_test_raw, y_test_raw = [], []
for i in tqdm(range(n), total=n, unit="measurements"):
    ind = i * 3
    X_raw.append([train.iloc[:,ind], train.iloc[:,ind+1], train.iloc[:,ind+2]])
    y_raw.append(meta['target'][ind])

for i in tqdm(range(n_test), total=n_test, unit="measurements"):
    ind = i * 3
    X_test_raw.append([test.iloc[:,ind], test.iloc[:,ind+1], test.iloc[:,ind+2]])


def transform_signal(x):
    signal_min = numpy.min(x)
    signal_max = numpy.max(x)
    return (signal_min, signal_max, numpy.mean(x), numpy.std(x), numpy.int16(signal_max) - signal_min)

def reduce_dimensionality(X, chunk_size=10000):
    X_transformed = []
    for i in tqdm(range(len(X)), total=len(X), unit="measurements"):
        record_measurements = []
        for chunk_num in range(int(len(X[0][0]) / chunk_size) - 1):
            fr = chunk_num * chunk_size
            to = (chunk_num + 1) * chunk_size
            record_measurements.append(list(transform_signal(X[i][0][fr:to])) + list(transform_signal(X[i][1][fr:to])) +  list(transform_signal(X[i][0][fr:to])))
        X_transformed.append(record_measurements)
    return numpy.asarray(X_transformed)


# X, y = reduce_dimensionality(X_raw, y_raw, chunk_size = 10000)

# reduced_dim = X.shape[1]
# measurement_features = X.shape[2]

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
    x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Attention(input_shape[1])(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def custom_fit(X, y):
    # splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=2019).split(X_transformed, y))
    splits = list(StratifiedShuffleSplit(n_splits=5, test_size=0.25).split(X, y))
    quality_list = numpy.zeros(20)
    for idx, (train_idx, val_idx) in enumerate(splits):
        K.clear_session() # I dont know what it do, but I imagine that it "clear session" :)
        print("Beginning fold {}".format(idx+1))
        train_X, train_y, val_X, val_y = X[train_idx], y[train_idx], X[val_idx], y[val_idx]
        model = model_lstm(train_X.shape)
        # ckpt = ModelCheckpoint('/srv/kkotochigov/weights_{}.h5'.format(idx), save_best_only=True, save_weights_only=True, verbose=1, monitor='val_matthews_correlation', mode='max')
        model.fit(train_X, train_y, batch_size=128, epochs=10, validation_data=[val_X, val_y])
        # model.load_weights('/srv/kkotochigov/weights_{}.h5'.format(idx))
        y_pred = model.predict(val_X, batch_size=512)
        quality = matthews_correlation(val_y, y_pred.reshape(y_pred.shape[0],))
        quality_list = numpy.add(quality_list, quality)
    quality_list = quality_list / 5
    print(quality_list)
    best_threshold = numpy.argmax(quality_list)
    return (5*best_threshold, quality_list[best_threshold])

for chunk_size in range(1000, 5000, 10000):
    # chunk_size = 10000
    X = reduce_dimensionality(X_raw,chunk_size)
    y = numpy.asarray(y_raw)
    # pandas.DataFrame(X).to_parquet(temp_dir+"/X_"+chunk_size+".parquet")
    X_test = reduce_dimensionality(X_test_raw, chunk_size)
    stats = custom_fit(X, y)


model = model_lstm(X.shape)
optimal_epochs = 50
model.fit(X, y, batch_size=128, epochs=optimal_epochs)
y_test = model.predict(X_test)

optimal_threshold = 0.4
y_test = y_test.reshape(len(y_test))
y_test = numpy.concatenate([[x,x,x] for x in y_test])
filename = "/srv/kkotochigov/vbs/submission_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
pandas.DataFrame({"signal_id": range(8712, 29049), "target":numpy.where(y_test > optimal_threshold, 1, 0)}).to_csv(filename, index=False)

# Submit solution
submission_text = "set 50 epochs"
os.system("kaggle competitions submit -f {} -m '{}' ".format(filename, submission_text))