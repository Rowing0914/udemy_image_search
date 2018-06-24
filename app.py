from keras import backend as K
from keras.datasets import mnist
from keras.models import model_from_json
import numpy as np
# heroku doesn't take _tinker, so this is a adhoc solution
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
  return render_template('index.html')

@app.route('/generateAgain')
def generateAgain():
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

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
