from models import *
from utilities import *
from keras.optimizers import *
from keras.layers import *
from keras.models import load_model
import tensorflow as tf
import h5py
import gc

def pipeline(base_model, model_name, fout_name, img_size, batch_size=32, n_ft_layers=-28):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(17, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # train top
    for layer in base_model.layers:
        layer.trainable = False
    model.summary()

    f2_val_score = train_model(model, model_name, Adam(1e-4), 50, batch_size=32, img_size=img_size)
    print ('Validation score: ', f2_val_score)
    gc.collect()

    # finetune deeper layers
    for layer in model.layers[n_ft_layers:]:
        layer.trainable = True
    for layer in model.layers[:n_ft_layers]:
        layer.trainable = False
    model.summary()
    print (len(model.layers))

    f2_val_score = train_model(model, model_name, Adam(1e-5), 10, batch_size=32, img_size=img_size)
    print ('Validation score: ', f2_val_score)
    gc.collect()
