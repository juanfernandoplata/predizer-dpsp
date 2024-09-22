import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.models import Model

def build_model():
    model = VGG16( weights = None )

    model.load_weights( "./vgg16.weights.h5" )

    model = Model( inputs = model.input, outputs = model.get_layer( "fc1" ).output )

    return model

def preprocess_img( img ):
    img = tf.io.decode_image( img )

    img = img[ tf.newaxis, ... ]

    img = tf.image.resize( img, ( 224, 224 ) )

    img = preprocess_input( img )

    return img
