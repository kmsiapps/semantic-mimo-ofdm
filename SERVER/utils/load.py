import json
import tensorflow as tf
from utils.log_config import getModelConfigs
from models.model import *
from models.cvc_block import SemViT_Decoder_Only, SemViT_Encoder_Only

__all__ = ['loadModel', 'loadJSCCModel', 'loadCLIPModel', 'loadUSRPModel', 'loadUSRPModelParam',\
    'loadREALModel', 'loadRICEModel', 'loadPOWERModel', 'loadSeparateModel']

def loadModel(model_name):
    model_configs = getModelConfigs()
    model_config = model_configs[model_name]
    model = globals()[model_config['model_class']](**model_config['model_params'])
    model(tf.zeros([1, 32, 32, 3]))
    model.load_weights(f'./logs/{model_name}/final').expect_partial()
    return model, model_config["caption"]
    