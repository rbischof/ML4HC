from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Input, Dropout, CuDNNLSTM, Masking, Bidirectional

class RNN:
	def __init__(self, name, output_dim, loss, last_activation, dense_size, hidden_size):
		self.name = name
		self.output_dim = output_dim
		self.loss = loss
		self.last_activation = last_activation
		self.dense_size = dense_size
		self.hidden_size = hidden_size
		self.model = self.get_model()
		self.compile_model()

	def get_model(self):
		x = Input(shape=(187, 1))
		layer = Masking(mask_value=0.0)
		lstm = Bidirectional(CuDNNLSTM(self.hidden_size), merge_mode='concat')(x)
		layer = Dense(self.dense_size, activation='relu')(lstm)
		layer = Dropout(.1)(layer)
		y = Dense(self.output_dim, name='out_layer', activation=self.last_activation)(layer)

		model = models.Model(inputs=x, outputs=y, name=self.name)
		return model

	def compile_model(self):
		self.model.compile(loss=self.loss, optimizer='Adam', metrics=['accuracy'])