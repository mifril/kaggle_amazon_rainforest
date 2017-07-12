from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers
from keras.applications.resnet50 import ResNet50
from metric import *


def build_model_1(in_size=256):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(in_size, in_size, 3)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='sigmoid'))
    # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', 'f2_score_keras'])
    model.summary()

    return model

def build_resnet(in_size=256):
    model = ResNet50(include_top=False, weights='imagenet', input_shape=(in_size, in_size, 3))

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model.add(top_model)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

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
