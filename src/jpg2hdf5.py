import numpy as np
import pandas as pd
import glob
import os
import h5py
from sklearn.model_selection import train_test_split

from utilities import *
from tqdm import tqdm
import cv2

RS = 17

def read_train_data(img_resize=256):
    x_train = []
    y_train = []

    df_train = pd.read_csv('../input/train_v2.csv')
    for f, tags in tqdm(df_train.values, miniters=1000):
        img = cv2.imread('C:\\data\\amazon\\' + 'train-jpg\\{}.jpg'.format(f))
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        x_train.append(cv2.resize(img, (img_resize, img_resize)))
        y_train.append(targets)
        
    return np.array(x_train, np.uint8), np.array(y_train, np.uint8)

def read_test_data(img_resize=256):
    x_test = []
    df_test = pd.read_csv('../input/sample_submission_v2.csv')
    for f, tags in tqdm(df_test.values):
        img = cv2.imread('C:\\data\\amazon\\test-jpg\\{}.jpg'.format(f))
        x_test.append(cv2.resize(img, (img_resize, img_resize)))
    return np.array(x_test, np.uint8)

def dump_dataset(path, X, y=None):
    with h5py.File(path, 'w') as f:
        f.create_dataset('X', data=X)
        if y is not None:
            f.create_dataset('y', data=y)

def jpg2hdf5(path, export_folder='C:\\data\\amazon\\', img_size=256):
    is_train = 'test' not in path
    if is_train:
        print ('train')
        X_train_all, y_train_all = read_train_data(img_size)
        X_train, X_val, y_train, y_val = train_val_split(X_train_all, y_train_all)

        save_path_train = export_folder + '/train.h5'
        save_path_val = export_folder + '/val.h5'
        # save_path_train_all = export_folder + '/train_all.h5'

        dump_dataset(save_path_train, X_train, y=y_train)
        dump_dataset(save_path_val, X_val, y=y_val)
        # dump_dataset(save_path_train_all, X_train_all, y=y_train_all)

        # with h5py.File(save_path_train, 'w') as f:
        #     f.create_dataset('X', data=X_train)
        #     f.create_dataset('y', data=y_train)

        # np.savez(save_path_train, X=X_train, y=y_train)
        # np.savez(save_path_train_all, X=X_train_all, y=y_train_all)
        # np.savez(save_path_val, X=X_val, y=y_val)
    else:
        print ('test')
        X_test = read_test_data(img_size)
        save_path_test = export_folder + '/test.h5'
        dump_dataset(save_path_test, X_test)
        # np.savez(save_path_train, X=X_test, f=files)

def jpg2hdf5_val_rescale(export_folder='C:\\data\\amazon\\', img_size=256):
    X_train_all, y_train_all = read_train_data(img_size)
    X_train, X_val, y_train, y_val = train_val_split(X_train_all, y_train_all)

    save_path_val = export_folder + '/val_{}_rescale.h5'.format(img_size)
    del X_train, y_train
    X_val = X_val.astype(np.float32)
    X_val *= 1. / 255.
    dump_dataset(save_path_val, X_val, y=y_val)

if __name__ == '__main__':
    # jpg2hdf5(path='C:\\data\\amazon\\train-jpg', img_size=224)
    # jpg2hdf5(path='C:\\data\\amazon\\test-jpg', img_size=224)
    jpg2hdf5_val_rescale('../features', img_size=256)
