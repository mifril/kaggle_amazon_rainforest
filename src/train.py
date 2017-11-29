import argparse
from pipeline import *

def train_model_1():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    pipeline(base_model, model_name ='resnet', fout_name='resnet', img_size=224, n_ft_layers=-28)

def train_model_2():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    pipeline(base_model, model_name ='resnet_256', fout_name='resnet_256', img_size=256, n_ft_layers=-28)

def train_model_4():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    pipeline(base_model, model_name ='resnet_256_2', fout_name='resnet_256_2', img_size=256, batch_size=32, n_ft_layers=-90)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument("--model", type=int, default=3, help="model to train")
    args = parser.parse_args()

    train_functions = [None, train_model_1, train_model_2, train_model_3]
    model_f = train_functions[args.model]
    model_f()
    gc.collect()
    
