import numpy as np
import pandas as pd
import glob
import os
import cv2
import h5py

from tqdm import tqdm

from metric import *
import tensorflow as tf
from keras.callbacks import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from keras.preprocessing.image import ImageDataGenerator, flip_axis
from keras.optimizers import *

FOLDER_INPUT = 'C:\\data\\amazon\\'
DATA_PATH = 'C:\\data\\amazon\\'
FEATURES_PATH = '../features/'
RS = 17

inv_label_map = ['blow_down',
 'bare_ground',
 'conventional_mine',
 'blooming',
 'cultivation',
 'artisinal_mine',
 'haze',
 'primary',
 'slash_burn',
 'habitation',
 'clear',
 'road',
 'selective_logging',
 'partly_cloudy',
 'agriculture',
 'water',
 'cloudy']

label_map = {'agriculture': 14,
 'artisinal_mine': 5,
 'bare_ground': 1,
 'blooming': 3,
 'blow_down': 0,
 'clear': 10,
 'cloudy': 16,
 'conventional_mine': 2,
 'cultivation': 4,
 'habitation': 9,
 'haze': 6,
 'partly_cloudy': 13,
 'primary': 7,
 'road': 11,
 'selective_logging': 12,
 'slash_burn': 8,
 'water': 15}

# based on https://www.kaggle.com/syeddanish/stratified-validation-split
def train_val_split(X_train_all, y_train_all):
    idx_train = []
    idx_val = []
    index = np.arange(X_train_all.shape[0])
    for i in tqdm(range(y_train_all.shape[1])):
        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=i)
        for train_index, test_index in sss.split(index, y_train_all[:,i]):
            idx_train_i, idx_test_i = index[train_index], index[test_index]
        # to ensure there is no repetetion within each split and between the splits
        idx_train = idx_train + list(set(list(idx_train_i)) - set(idx_train) - set(idx_val))
        idx_val = idx_val + list(set(list(idx_test_i)) - set(idx_val) - set(idx_train))
    return X_train_all[idx_train], X_train_all[idx_val], y_train_all[idx_train], y_train_all[idx_val]

def read_y_train():
    y_train = []
    df_train = pd.read_csv('../input/train_v2.csv')

    flatten = lambda l: [item for sublist in l for item in sublist]
    y = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

    for f, tags in tqdm(df_train.values, miniters=1000):
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        y_train.append(targets)

    return np.array(y_train, np.uint8)

def train_top_model(model, x_train, y_train, model_name, opt, n_epoch=100, batch_size=128):
    X_train, X_val, y_train, y_val = train_val_split(x_train, y_train)

    m = 'val_loss'
    early_stopping = EarlyStopping(monitor=m, patience=2, verbose=1, mode='auto')
    filename = 'weights_' + str(model_name) + '/{val_loss:.5f}-{epoch:03d}.h5'
    checkpoint = ModelCheckpoint(filename, monitor=m, verbose=1, save_best_only=True)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    wdir = 'weights_' + str(model_name) + '/'
    print('Search weights: ', wdir)

    if os.path.exists(wdir) and len(os.listdir(wdir)) > 0:
        wf = sorted(glob.glob(wdir + '*.h5'))[0]
        model.load_weights(wf)
        print('loaded weights file: ', wf)

    if not os.path.exists(wdir):
        os.mkdir(wdir)

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    model.fit(X_train, y_train, batch_size=batch_size,
                        epochs=n_epoch,
                        verbose=1,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping, checkpoint])
    
    preds_val = model.predict(X_val)
    pd.DataFrame(preds_val).to_csv('../X/preds_val_{}.csv'.format(model_name), index=None)
    f2_val_score = f2_score(y_val, np.array(preds_val) > 0.2)

    return f2_val_score

def get_test_batch(X, tta=False, batch_size=32):
    if not tta:
        i = 0
        while i*batch_size < X.shape[0]:
            batch_x = X[i*batch_size:(i+1)*batch_size].astype(K.floatx()) if (i+1)*batch_size < X.shape[0] else X[i*batch_size:].astype(K.floatx())
            batch_x *= 1. / 255.
            i += 1
            yield batch_x
    else:
        for img in X:
            if X.shape[1] == 256:
                batch_x = np.zeros((12, X.shape[1], X.shape[2], 3), dtype=K.floatx())
                idx = 0
                for aug_img in [img, np.rot90(img, 1), np.rot90(img, 2), np.rot90(img, 3)]:
                    for axis in (None, 0, 1):
                        if axis is None:
                            flipped_img = img
                        else:
                            flipped_img = flip_axis(img, axis)
                        batch_x[idx] = flipped_img
                        idx += 1
            elif X.shape[1] == 224:
                batch_x = np.zeros((12, X.shape[1], X.shape[2], 3), dtype=K.floatx())
                batch_x *= 1. / 255.
                idx = 0
                for crop in (
                        np.s_[32:, 32:, :],
                        np.s_[:224, :224, :],
                        np.s_[32:, :224, :],
                        np.s_[:224, 32:, :],
                ):
                    cropped_img = img[crop]
                    for axis in (None, 0, 1):
                        if axis is None:
                            flipped_img = cropped_img
                        else:
                            flipped_img = flip_axis(cropped_img, axis)
                        batch_x[idx] = flipped_img
                        idx += 1
            batch_x *= 1. / 255.
            yield batch_x

def train_model(model, model_name, opt, n_epoch=100, batch_size=128, es_patience=1, img_size=256):
    if img_size == 224:
        f_train = h5py.File(os.path.join(DATA_PATH, 'train.h5'))
        f_val = h5py.File(os.path.join(DATA_PATH, 'val.h5'))
    else:
        f_train = h5py.File(os.path.join(FEATURES_PATH, 'train_' + str(img_size) + '.h5'))
        if os.path.exists(os.path.join(FEATURES_PATH, 'val_' + str(img_size) + '_rescale.h5')):
            f_val = h5py.File(os.path.join(FEATURES_PATH, 'val_' + str(img_size) + '_rescale.h5'))
        else:
            f_val = h5py.File(os.path.join(FEATURES_PATH, 'val_' + str(img_size) + '.h5'))
    
    X_train = f_train['X']
    y_train = f_train['y']
    X_val = f_val['X']
    y_val = f_val['y']

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    # early_stopping = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=0, mode='auto')
    filename = 'weights_' + str(model_name) + '/{val_loss:.5f}-{epoch:03d}.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True)
    
    def scheduler(epoch):
        if epoch % 10 == 0 and epoch != 0:
            lr = K.gte_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr*.1)
            print("lr changed to {}".format(lr*.1))
        return K.get_value(model.optimizer.lr)
    change_lr = LearningRateScheduler(scheduler)
    
    wdir = 'weights_' + str(model_name) + '/'
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    elif len(os.listdir(wdir)) > 0:
        wf = sorted(glob.glob(wdir + '*.h5'))[0]
        model.load_weights(wf)
        print('loaded weights file: ', wf)

    datagen = ImageDataGenerator(
            rescale=1./255.,
            zoom_range=0.15,
            rotation_range=90.,
            horizontal_flip=True,
            vertical_flip=True)
    
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        epochs=n_epoch,
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        verbose=1,
                        validation_data=(X_val, y_val),
                        callbacks=[checkpoint, change_lr])

    preds_val = model.predict(X_val, batch_size=batch_size)
    pd.DataFrame(preds_val).to_csv('../features/preds_val_{}.csv'.format(model_name), index=None)
    f2_val_score = f2_score(y_val, np.array(preds_val) > 0.2)

    return f2_val_score

def predict(model, model_name, fout_name, batch_size, img_size=256, tta=True):
    if img_size == 224:
        f_val = h5py.File(os.path.join(DATA_PATH, 'val.h5'))
    else:
        if os.path.exists(os.path.join(FEATURES_PATH, 'val_' + str(img_size) + '_rescale.h5')):
            f_val = h5py.File(os.path.join(FEATURES_PATH, 'val_' + str(img_size) + '_rescale.h5'))
        else:
            f_val = h5py.File(os.path.join(FEATURES_PATH, 'val_' + str(img_size) + '.h5'))

    X_val = f_val['X']
    y_val = f_val['y']
    
    wdir = 'weights_' + str(model_name) + '/'
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    elif len(os.listdir(wdir)) > 0:
        wf = sorted(glob.glob(wdir + '*.h5'))[0]
        model.load_weights(wf)
        print('loaded weights file: ', wf)

    best_threshold = [0.2]*17
    preds_val = model.predict(X_val, batch_size=batch_size)
    f2_val_score = f2_score(y_val, np.array(preds_val) > 0.2)
    print ('f2_score: ', f2_val_score)
    best_threshold, best_f2_score = get_optimal_threshhold(y_val, preds_val, iterations = 100)
    print ('Optimal threshold: ', best_threshold, ', f2_score: ', best_f2_score)

    if img_size == 224:
        f_test = h5py.File(os.path.join(DATA_PATH, 'test.h5'))
    else:
        f_test = h5py.File(os.path.join(FEATURES_PATH, 'test_' + str(img_size) + '.h5'))

    X_test = f_test['X']

    generator = get_test_batch(X_test, tta=tta)
    preds_test = []
    preds_test_all = []
    for batch in tqdm(generator):
        batch_predictions = model.predict(batch)
        if tta:
            preds_test.append(batch_predictions.mean(axis=0))
            preds_test_all.append(batch_predictions)
        else:
            preds_test.append(batch_predictions)

    if tta:
        preds_test_all = np.array(preds_test_all)
        preds_test_all = preds_test_all.reshape(preds_test_all.shape[0], preds_test_all.shape[1] * preds_test_all.shape[2])
        print (preds_test_all.shape)
        print (np.array(preds_test).shape)
    else:
        preds_test = np.concatenate(preds_test)

    tags_test = []
    for pred in tqdm(preds_test):
        tags = [inv_label_map[j] for j in range(len(pred)) if pred[j] > best_threshold[j]]
        tags_test.append(' '.join(tags))

    df_test = pd.read_csv('../input/sample_submission_v2.csv')
    df_test['tags'] = tags_test
    df_test.to_csv('../output/{}.csv'.format(fout_name), index=False)
    if tta:
        pd.DataFrame(preds_test).to_csv('../output/preds_{}_tta2.csv'.format(fout_name), index=False)
        pd.DataFrame(preds_test_all).to_csv('../output/preds_{}_all_tta2.csv'.format(fout_name), index=False)
