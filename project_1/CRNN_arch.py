from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, CuDNNGRU, BatchNormalization, Bidirectional
		
class CRNN:
    def __init__(self, name, output_dim, loss, last_activation, dense_size, hidden_size):
        self.name = name
        self.output_dim = output_dim
        self.loss = loss
        self.dense_size = dense_size
        self.hidden_size = hidden_size
        self.last_activation = last_activation
        self.model = self.get_model()
        self.compile_model()
  
    def get_model(self):
        x = Input(shape=(187, 1))
        layer = Convolution1D(16, kernel_size=3, activation='relu', padding="valid")(x)
        layer = Convolution1D(16, kernel_size=3, activation='relu', padding="valid")(layer)
        layer = MaxPool1D(pool_size=2)(layer)
        layer = Dropout(rate=0.1)(layer)
        layer = Convolution1D(32, kernel_size=3, activation='relu', padding="valid")(layer)
        layer = Convolution1D(32, kernel_size=3, activation='relu', padding="valid")(layer)
        layer = MaxPool1D(pool_size=2)(layer)
        layer = Dropout(rate=0.1)(layer)
        layer = BatchNormalization()(layer)
        gru = Bidirectional(CuDNNGRU(self.hidden_size, name='rnn'), merge_mode='concat')(layer)
        layer = Dense(self.dense_size, activation='relu', name='dense')(gru)
        y = Dense(self.output_dim, name='out_layer', activation=self.last_activation)(layer)

        model = models.Model(inputs=x, outputs=y, name=self.name)
        return model

    def compile_model(self):
        self.model.compile(loss=self.loss, optimizer='Adam', metrics=['accuracy'])