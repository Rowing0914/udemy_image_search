import keras
from keras.datasets import mnist
from keras.models import model_from_json
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

# heroku doesn't take _tinker, so this is a adhoc solution...
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
  return render_template('index.html')

@app.route('/trainAgain')
def trainAgain():
	img_rows = 28
	img_cols = 28
	checker = True
	index_range = 30

	# load json and create model
	json_file = open('./models/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("./models/model.h5")

	while (checker):
		# the data, split between train and test sets
		(_, _), (x_test, y_test) = mnist.load_data()
		index = np.random.randint(len(x_test))
		x_test = x_test[index:index+index_range]
		y_test = y_test[index:index+index_range]

		if K.image_data_format() == 'channels_first':
			x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
			input_shape = (1, img_rows, img_cols)
		else:
			x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
			input_shape = (img_rows, img_cols, 1)

		x_test = x_test.astype('float32')
		x_test /= 255
		integers = set()
		fig = plt.figure()

		for i in range(index_range):
			a = loaded_model.predict(x_test[i].reshape(1, 28, 28, 1))
			integers.add(np.argmax(a))
			plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
			fig.savefig('./static/images/{}.png'.format(np.argmax(a)))

		if integers == set((0,1,2,3,4,5,6,7,8,9)):
			checker = False
	return render_template('index.html')

@app.route('/a')
def a():
	batch_size = 128
	num_classes = 10
	epochs = 5

	# input image dimensions
	img_rows, img_cols = 28, 28

	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
		x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
					 activation='relu',
					 input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
				  optimizer=keras.optimizers.Adadelta(),
				  metrics=['accuracy'])

	model.fit(x_train, y_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  verbose=1,
			  validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, verbose=0)

	# save a model architecture
	from keras.utils import plot_model
	plot_model(model, to_file='model.png')

	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	# serialize model to JSON
	model_json = model.to_json()
	with open("./models/model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("./models/model.h5")
	print("Saved model to disk")
	return "done"

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
