import keras
from keras.datasets import mnist
from keras.models import model_from_json
from keras import backend as K
import numpy as np
from PIL import Image, ImageOps
# import matplotlib.pyplot as plt
from scipy.misc import toimage


class img_saver():
  def __init__(self):
    # input image dimensions
    self.img_rows = 28
    self.img_cols = 28
    self.checker = True
    self.index_range = 30
    
  def main(self):
    # load json and create model
    json_file = open('../models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("../models/model.h5")

    while (self.checker):
      # the data, split between train and test sets
      (_, _), (x_test, _) = mnist.load_data()
      index = np.random.randint(len(x_test))
      x_test = x_test[index:index+self.index_range]

      if K.image_data_format() == 'channels_first':
          x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
          input_shape = (1, self.img_rows, self.img_cols)
      else:
          x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
          input_shape = (self.img_rows, self.img_cols, 1)

      x_test = x_test.astype('float32')
      x_test /= 255
      
      integers = set()
      # fig = plt.figure()

      for i in range(self.index_range):
        a = loaded_model.predict(x_test[i].reshape(1, 28, 28, 1))
        integers.add(np.argmax(a))
        A = x_test[i].reshape(28, 28)
        im = toimage(A)
        im.save('../static/images/{}.png'.format(np.argmax(a)))
        # plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        # fig.savefig('./static/images/{}.png'.format(np.argmax(a)))
      
      if integers == set((0,1,2,3,4,5,6,7,8,9)):
        self.checker = False

if __name__ == '__main__':
  img_saver = img_saver()
  img_saver.main()