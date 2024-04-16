# importing necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# deep learning modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
from tensorflow.keras import regularizers
# data preprocessing
from keras.utils import to_categorical


def load_data():
    # load the dataset and split into training/testing features and labels
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # data normalization
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data()

# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(x_train[i]) 
# show the figure
plt.show()

# Label encoding
# one-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# print first ten (one hot) training labels
print('One-hot labels: ')
print(y_train[:10])


def define_model():
    # Defines a simple neural network architecture for MNIST classification
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()
    # compile and run
    model.compile(optimizer='Adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    return model


model = define_model()


import numpy as np


def evaluate_test_accuracy(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    accuracy = 100 * score[1]
    print('Test accuracy: %.2f%%' % accuracy)


def make_prediction(model, x_test, y_test, index):
    image = x_test[index]
    yhat = model.predict(np.asarray([image]))
    predicted_class = np.argmax(yhat)
    true_class = np.argmax(y_test[index])
    print('Predicted values:', yhat)
    print('\nPredicted class:', predicted_class)
    print('True class:', true_class)


evaluate_test_accuracy(model, x_test, y_test)
make_prediction(model, x_test, y_test, 102)


# checking all layers present in the model
for layer in model.layers:
    print(layer.name)  

# summarize filter shapes
for layer in model.layers:
    # check for convolutional layer
    if 'conv' not in layer.name:
        continue
    # get filter weights
    filters, biases = layer.get_weights()
    print(layer.name, filters.shape)

# retrieve weights from the second hidden layer
filters, biases = model.layers[0].get_weights()
print(filters.shape, biases.shape)

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

from matplotlib import pyplot

# plot first few filters
n_filters, ix = 15, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(1):
		# specify subplot and turn off axis
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in greyscale
		pyplot.imshow(f[:, :, j], cmap='gray')
		ix += 1
# show the figure
pyplot.show()

from keras.models import Model

# layer_outputs should match the number of Conv2D and MaxPooling2D blocks in the model
layer_outputs = [layer.output for layer in model.layers[0:4]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

layer_outputs


activation_model.summary()

img = x_test[51].reshape(1,28,28,1)
fig = plt.figure(figsize=(5,5))
plt.imshow(img[0,:,:,0],cmap="gray")
plt.axis('off')

activations = activation_model.predict(img)


# Grad-cam


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )


    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)


    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
     

def plt_images(x_test, i_max, j_max):

  fig, axs = plt.subplots(i_max, j_max, figsize=plt.figaspect(0.5))
  for i in range(i_max):
    for j in range(j_max):
      ind = np.arange(i_max*j_max)[i*j_max+j]
      img = x_test[ind].reshape(1,28,28,1)
      axs[i, j].imshow(img[0,:,:,0], aspect='auto')

  return
i_max = 3
j_max = 5
plt_images(x_train, i_max, j_max)    
plt.show()

def plot_gradcam_images(i_max, j_max, x_test, model, layer_name):
  fig, axs = plt.subplots(i_max, j_max, figsize=plt.figaspect(0.5))
  for i in range(i_max):
    for j in range(j_max):
      ind = np.arange(i_max*j_max)[i*j_max+j]
      heatmap = make_gradcam_heatmap(tf.expand_dims(x_test[ind], axis=0), model, layer_name)
      axs[i, j].matshow(heatmap, aspect='auto')
  plt.tight_layout()
  plt.show()
  return

model.summary()




model = define_model()  


# Now proceed with visualizing Grad-CAM heatmaps
plot_gradcam_images(i_max, j_max, x_train, model, 'conv2d_1')

plot_gradcam_images(i_max, j_max, x_train, model, 'max_pooling2d_1')