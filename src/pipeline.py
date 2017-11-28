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

    predict(model, model_name, fout_name=fout_name, batch_size=batch_size, img_size=img_size, tta=False)
    predict(model, model_name, fout_name=fout_name + '_tta', batch_size=batch_size, img_size=img_size, tta=True)

def model_1():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    pipeline(base_model, model_name ='resnet', fout_name='resnet', img_size=224, n_ft_layers=-28)

def model_2():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    pipeline(base_model, model_name ='resnet_256', fout_name='resnet_256', img_size=256, n_ft_layers=-28)

def model_4():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    pipeline(base_model, model_name ='resnet_256_2', fout_name='resnet_256_2', img_size=256, batch_size=32, n_ft_layers=-90)

if __name__ == '__main__':
    model_4()
    gc.collect()
