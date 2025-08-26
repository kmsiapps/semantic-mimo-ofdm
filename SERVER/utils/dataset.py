import tensorflow as tf
import numpy as np
import pickle

def loadCifarDataset(train_bs, test_bs, root, type='all', mini=False):
    datasets = []
    
    if type in ['train', 'all']:
        train_ds = []
        for i in range(5):
            with open(root + f'data_batch_{i+1}', 'rb') as f_train:
                dict = pickle.load(f_train, encoding='bytes')
                train_ds.append(np.reshape(dict[b'data'], (-1, 3, 32, 32)).transpose((0, 2, 3, 1)))
        train_ds = np.vstack(train_ds)
        if mini:
            train_ds = train_ds[0:200]
        train_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_ds, tf.float32))
        normalize_and_augment = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            ])
        train_ds = (
            train_ds.batch(train_bs)
                    .cache()
                    .shuffle(50000, reshuffle_each_iteration=True)
                    .map(lambda x: normalize_and_augment(x, training=True), num_parallel_calls=tf.data.AUTOTUNE)
                    .map(lambda x: (x, x))
                    .prefetch(tf.data.AUTOTUNE)
        )
        datasets.append(train_ds)
    
    if type in ['test', 'all']:
        with open(root + f'/test_batch', 'rb') as f_test:
            dict = pickle.load(f_test, encoding='bytes')
            test_ds = np.reshape(dict[b'data'], (-1, 3, 32, 32)).transpose((0, 2, 3, 1))
        if mini:
            test_ds = test_ds[0:300]
        test_ds = tf.data.Dataset.from_tensor_slices(tf.cast(test_ds, tf.float32))
        normalize = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        test_ds = (
            test_ds.batch(test_bs)
                   .map(lambda x: normalize(x))
                   .map(lambda x: (x, x))
                   .cache()
                   .prefetch(tf.data.AUTOTUNE)
        )
        datasets.append(test_ds)
            
    return tuple(datasets)


def loadCifarValidImage(root, image_number):
    with open(root + f'/test_batch', 'rb') as f_test:
        dict = pickle.load(f_test, encoding='bytes')
        test_ds = np.reshape(dict[b'data'], (-1, 3, 32, 32)).transpose((0, 2, 3, 1)) / 255.
    test_ds = tf.expand_dims(tf.cast(test_ds[image_number], tf.float32), axis=0) 
    
    return test_ds

def loadCifarValidImages(root, image_number, size=10):
    with open(root + f'/test_batch', 'rb') as f_test:
        dict = pickle.load(f_test, encoding='bytes')
        test_ds = np.reshape(dict[b'data'], (-1, 3, 32, 32)).transpose((0, 2, 3, 1)) / 255.
    test_ds = tf.cast(test_ds[image_number:image_number+size], tf.float32)
    
    return test_ds