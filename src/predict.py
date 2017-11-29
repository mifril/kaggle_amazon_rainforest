from models import *
from utilities import *
from keras.optimizers import *
from keras.layers import *
from keras.models import load_model
import tensorflow as tf
import h5py
import gc

def predict_model_4():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3))

    model_name ='resnet_256_2'
    fout_name='resnet_256_2'
    img_size=256

    predict(model, model_name, fout_name=fout_name, batch_size=batch_size, img_size=img_size, tta=False)
    predict(model, model_name, fout_name=fout_name + '_tta', batch_size=batch_size, img_size=img_size, tta=True)

if __name__ == '__main__':
    predict_model_4()
    gc.collect()
