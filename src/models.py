from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers
from keras.applications.resnet50 import ResNet50
from metric import *

def build_resnet_topmodel_no_dense(in_shape, resnet_in_size=256):
    top_model = Sequential()
    top_model.add(Dense(17, activation='sigmoid', input_shape=in_shape))
    top_model.summary()

    return top_model

def build_resnet_topmodel(in_shape, resnet_in_size=256):
    top_model = Sequential()
    top_model.add(Dense(256, activation='relu', input_shape=in_shape))
    top_model.add(Dropout(0.2))
    top_model.add(Dense(17, activation='sigmoid'))
    top_model.summary()

    return top_model

def build_resnet(top_model, in_size=256):
    resnet_model = ResNet50(include_top=False, weights='imagenet', input_shape=(in_size, in_size, 3))
    x = resnet_model.output
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    for layer in top_model.layers:
        x = layer(x)

    final_model = Model(resnet_model.input, x)

    for layer in final_model.layers[-28:]:
        layer.trainable = True
    for layer in final_model.layers[:-28]:
        layer.trainable = False

    final_model.summary()

    return final_model
