from models import *
from utilities import *
from keras.optimizers import *
from keras.layers import *
from keras.models import load_model
import tensorflow as tf
import h5py
import gc

# def predict_base(img_size, model, model_name, fout_name):
#     with tf.device('/gpu:0'):
#         if img_size != 224:
#             f_train = h5py.File(os.path.join(FEATURES_PATH, 'train_' + str(img_size) + '.h5'))
#             f_val = h5py.File(os.path.join(FEATURES_PATH, 'val_' + str(img_size) + '.h5'))
#         else:
#             f_train = h5py.File(os.path.join(DATA_PATH, 'train.h5'))
#             f_val = h5py.File(os.path.join(DATA_PATH, 'val.h5'))
#         X_train = f_train['X']
#         X_val = f_val['X']

#         preds_train = model.predict(X_train)
#         preds_val = model.predict(X_val)
#         preds_train = preds_train.reshape((preds_train.shape[0], preds_train.shape[1]*preds_train.shape[2]*preds_train.shape[3]))
#         preds_val = preds_val.reshape((preds_val.shape[0], preds_val.shape[1]*preds_val.shape[2]*preds_val.shape[3]))

#         pd.DataFrame(np.concatenate((preds_train, preds_val), axis=0)).to_csv(fout_name, index=None)

# def train_top(img_size, fpreds_base, model_name, epochs_arr, opts, batch_size=128, top_model_no_dense=False):
#     x_train = pd.read_csv(fpreds_base).values
#     y_train = read_y_train()
    
#     if top_model_no_dense:
#         top_model = build_resnet_topmodel_no_dense(in_shape = x_train.shape[1:], resnet_in_size=img_size)
#     else:
#         top_model = build_resnet_topmodel(in_shape = x_train.shape[1:], resnet_in_size=img_size)

#     for opt, e in zip(opts, epochs_arr):
#         f2_val_score = train_top_model(top_model, x_train, y_train, model_name, opt, e, batch_size)
#         print ('Validation score: ', f2_val_score)

#     return top_model

# def finetune(img_size, model_name, top_model, epochs_arr, opts, batch_size=32):
#     model = build_resnet(top_model, 224)
   
#     for opt, e in zip(opts, epochs_arr):
#         f2_val_score = train_model(model, model_name, opt, e, batch_size)
#         print ('Validation score: ', f2_val_score)
#         gc.collect()

#     model.save_weights('weights_' + model_name + '/{:.5}-final.h5'.format(f2_val_score))
#     model.save('weights_' + model_name + '/model.h5')
    
#     if img_size != 224:
#         f_val = h5py.File(os.path.join(FEATURES_PATH, 'val_' + str(img_size) + '.h5'))
#     else:
#         f_val = h5py.File(os.path.join(DATA_PATH, 'val.h5'))
#     X_val = f_val['X']
#     y_val = f_val['y']
#     preds_val = model.predict(X_val)
#     best_threshold, best_f2_score = get_optimal_threshhold(y_val, preds_val, iterations = 100)

#     print ('Optimal threshold: ', best_threshold, ', f2_score: ', best_f2_score)

#     return model, best_threshold

# def predict(img_size, model, best_threshold, fout_name):
#     with tf.device('/gpu:0'):
#         if img_size != 224:
#             f_test = h5py.File(os.path.join(FEATURES_PATH, 'test_' + str(img_size) + '.h5'))
#         else:
#             f_test = h5py.File(os.path.join(DATA_PATH, 'test.h5'))
#         X_test = f_test['X']

#         preds_test = model.predict(X_test)
        
#         tags_test = []
#         for pred in tqdm(preds_test):
#             tags = [inv_label_map[j] for j in range(len(pred)) if pred[j] > best_threshold[j]]
#             tags_test.append(' '.join(tags))

#         df_test = pd.read_csv('../input/sample_submission_v2.csv')
#         df_test['tags'] = tags_test
#         df_test.to_csv('../output/' + fout_name + '.csv', index=False)

# def train_model_1():
#     with tf.device('/gpu:0'):
#         img_size = 256
#         top_model_name = 'resnet_top'
#         epochs_arr = [1000, 500, 500]
#         opts = [Adam(1e-3), Adam(1e-4), Adam(1e-5)]
#         batch_size = 128
#         top_model = train_top(img_size, '../features/preds_train_ResNet50_256.csv', model_name, epochs_arr, opts, batch_size=128)

#         x_train = pd.read_csv('../features/preds_train_ResNet50_256.csv').values
#         y_train = read_y_train()
#         X_train, X_val, y_train, y_val = train_val_split(x_train, y_train)

#         preds_val = model.predict(X_val)
#         print ('f2 val: ', f2_score(y_val, np.array(preds_val) > 0.2))
#         best_threshold, best_f2_score = get_optimal_threshhold(y_val, preds_val, iterations = 100)
#         print ('f2 val (best threshold): ', best_f2_score)

#         X_test = pd.read_csv('../features/preds_test_ResNet50_256.csv').values
#         preds_test = model.predict(X_test)
        
#         tags_test = []
#         for pred in tqdm(preds_test):
#             tags = [inv_label_map[j] for j in range(len(pred)) if pred[j] > best_threshold[j]]
#             tags_test.append(' '.join(tags))

#         df_test = pd.read_csv('../input/sample_submission_v2.csv')
#         df_test['tags'] = tags_test
#         df_test.to_csv('../output/resnet50_256_top.csv', index=False)

# def train_model_2():
#     with tf.device('/gpu:0'):
#         img_size = 256
#         top_model_name = 'resnet_top'
#         epochs_arr = [1000, 500, 500]
#         opts = [Adam(1e-3), Adam(1e-4), Adam(1e-5)]
#         batch_size = 128
#         top_model = train_top(img_size, '../features/preds_train_ResNet50_256.csv', top_model_name, epochs_arr, opts, batch_size=128)

#         model_name = 'resnet_adam'
#         epochs_arr = [100, 50, 50]
#         opts = [Adam(1e-4), Adam(1e-5), SGD(1e-6, momentum=0.9, nesterov=True)]
#         batch_size = 32
#         model, best_threshold = finetune(img_size, model_name, top_model, epochs_arr, opts, batch_size=32)

#         predict(img_size, model, best_threshold, fout_name='resnet50')

# def train_model_3():
#     with tf.device('/gpu:0'):
#         img_size = 224
#         model_name = 'resnet50'
#         top_model_name = 'resnet50_top'
#         fpreds_base = '../features/preds_train_{}.csv'.format(model_name)

#         if not os.path.exists(fpreds_base):
#             base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
#             predict_base(img_size, base_model, model_name, fpreds_base)
        
#         epochs_arr = [1000, 500, 500]
#         opts = [Adam(1e-2), Adam(1e-3), Adam(1e-4), Adam(1e-5), Adam(1e-6)]
#         batch_size = 128
#         top_model = train_top(img_size, fpreds_base, top_model_name, epochs_arr, opts, batch_size=128)

#         epochs_arr = [100, 50, 50]
#         opts = [Adam(1e-4), Adam(1e-5), SGD(1e-6, momentum=0.9, nesterov=True)]
#         batch_size = 32
#         model, best_threshold = finetune(img_size, model_name, top_model, epochs_arr, opts, batch_size=32)

#         predict(img_size, model, best_threshold, fout_name='resnet50')

# # def train_model_4():
# #     with tf.device('/gpu:0'):
# #         img_size = 224
# #         model_name = 'resnet50_no_dense'
# #         top_model_name = 'resnet50_no_dense_top'
# #         fpreds_base = '../features/preds_train_resnet50.csv'

# #         if not os.path.exists(fpreds_base):
# #             base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
# #             predict_base(img_size, base_model, model_name, fpreds_base)
        
# #         epochs_arr = [1000, 500, 500]
# #         opts = [Adam(1e-3), Adam(1e-4), Adam(1e-5), Adam(1e-6), Adam(1e-7)]
# #         batch_size = 128
# #         top_model = train_top(img_size, fpreds_base, top_model_name, epochs_arr, opts, batch_size=128, top_model_no_dense=True)

# #         epochs_arr = [100, 50, 50]
# #         opts = [Adam(1e-4), Adam(1e-5), Adam(1e-6), SGD(1e-6, momentum=0.9)]
# #         batch_size = 32
# #         model, best_threshold = finetune(img_size, model_name, top_model, epochs_arr, opts, batch_size=32)

# #         predict(img_size, model, best_threshold, fout_name='resnet50_no_dense')

# def train_model_5():
#     with tf.device('/gpu:0'):
#         img_size = 256

#         model_name = 'resnet50_crop'
#         top_model_name = 'resnet50_crop_top'
#         fpreds_base = '../features/preds_train_resnet50.csv'.format(model_name)

#         if not os.path.exists(fpreds_base):
#             base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
#             predict_base(224, base_model, model_name, fpreds_base)

#         epochs_arr = [1000, 500, 500]
#         opts = [Adam(1e-3), Adam(1e-4), Adam(1e-5)]
#         batch_size = 128
#         top_model = train_top(224, '../features/preds_train_resnet50.csv', top_model_name, epochs_arr, opts, batch_size=128)

#         model_name = 'resnet_adam'
#         epochs_arr = [100, 50, 50]
#         opts = [Adam(1e-3), Adam(1e-4), Adam(1e-5), Adam(1e-6), SGD(1e-6, momentum=0.9)]
#         batch_size = 32
#         model, best_threshold = finetune(256, model_name, top_model, epochs_arr, opts, batch_size=32)

#         predict(img_size, model, best_threshold, fout_name='resnet50')


def pipeline(base_model, model_name, fout_name, img_size, batch_size=32, n_ft_layers=-28):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(17, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # # train top
    # for layer in base_model.layers:
    #     layer.trainable = False
    # model.summary()

    # epochs_arr = [100, 100]
    # opts = [Adam(1e-4), Adam(1e-5)]
    # es = 3
    # for opt, e in zip(opts, epochs_arr):
    #     f2_val_score = train_model(model, model_name, opt, e, batch_size, es, img_size=img_size)
    #     print ('Validation score: ', f2_val_score)
    #     gc.collect()

    # # finetune deeper layers
    # for layer in model.layers[n_ft_layers:]:
    #     layer.trainable = True
    # for layer in model.layers[:n_ft_layers]:
    #     layer.trainable = False
    # model.summary()
    # print (len(model.layers))

    # epochs_arr = [100, 100, 100]
    # opts = [SGD(1e-6, momentum=True), SGD(1e-7, momentum=True), SGD(1e-8, momentum=True)]
    # es = 1
    # for opt, e in zip(opts, epochs_arr):
    #     f2_val_score = train_model(model, model_name, opt, e, batch_size, es, img_size=img_size)
    #     print ('Validation score: ', f2_val_score)
    #     gc.collect()

    # train top
    # for layer in base_model.layers:
    #     layer.trainable = False
    # model.summary()

    # f2_val_score = train_model(model, model_name, Adam(1e-4), 50, batch_size=32, img_size=img_size)
    # print ('Validation score: ', f2_val_score)
    # gc.collect()

    # # finetune deeper layers
    # for layer in model.layers[n_ft_layers:]:
    #     layer.trainable = True
    # for layer in model.layers[:n_ft_layers]:
    #     layer.trainable = False
    # model.summary()
    # print (len(model.layers))

    # f2_val_score = train_model(model, model_name, Adam(1e-5), 10, batch_size=32, img_size=img_size)
    # print ('Validation score: ', f2_val_score)
    # gc.collect()

    # predict(model, model_name, fout_name=fout_name, batch_size=batch_size, img_size=img_size, tta=False)
    predict(model, model_name, fout_name=fout_name + '_tta', batch_size=batch_size, img_size=img_size, tta=True)

def model_1():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    pipeline(base_model, model_name ='resnet', fout_name='resnet', img_size=224, n_ft_layers=-28)

def model_2():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    pipeline(base_model, model_name ='resnet_256', fout_name='resnet_256', img_size=256, n_ft_layers=-28)

def model_3():
    from densenet121 import densenet121_model
    base_model = densenet121_model(img_rows=256, img_cols=256, color_type=3, dropout_rate=0.2)
    pipeline(base_model, model_name ='densenet_121', fout_name='densenet_121', img_size=256, batch_size=32, n_ft_layers=-360)

def model_4():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    pipeline(base_model, model_name ='resnet_256_2', fout_name='resnet_256_2', img_size=256, batch_size=32, n_ft_layers=-90)

if __name__ == '__main__':
    model_4()
    gc.collect()
