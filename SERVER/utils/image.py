import tensorflow as tf
import numpy as np
import PIL.Image as pilimg

def imBatchtoImage(batch_images):
    '''
    turns b, 32, 32, 3 images into single sqrt(b) * 32, sqrt(b) * 32, 3 image.
    '''
    batch, h, w, c = batch_images.shape
    b = int(batch ** 0.5)

    divisor = b
    while batch % divisor != 0:
        divisor -= 1
    
    image = tf.reshape(batch_images, (-1, batch//divisor, h, w, c))
    image = tf.transpose(image, [0, 2, 1, 3, 4])
    image = tf.reshape(image, (-1, batch//divisor*w, c))
    return image

def loadSampleImage(img_path=f'images/sample_img.png'):
    origin_images = pilimg.open(img_path).convert('RGB')

    # Encode image
    images = tf.convert_to_tensor(np.array(origin_images), dtype=tf.float32) / 255.0
    h, w, c = images.shape
    images = tf.reshape(images, (1, h, w, c))

    images = tf.image.extract_patches(
        images,
        sizes=[1, 32, 32, 1],
        strides=[1, 32, 32, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    images = tf.reshape(images, (-1, 32, 32, c))
    return images
