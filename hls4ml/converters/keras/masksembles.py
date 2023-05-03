import math
from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler
from hls4ml.converters.utils import parse_data_format, compute_padding_1d, compute_padding_2d
from hls4ml.converters.keras.core import BinaryQuantizer

@keras_handler('Masksembles')
def parse_masksembles_layer(keras_layer, input_names, input_shapes, data_reader):
    assert('Masksembles' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)
    weights_shape = data_reader.get_weights_shape(layer['name'], 'kernel')
    layer['n_in'] = weights_shape[0]
    layer['num_masks'] = keras_layer['config']['num_masks']
    layer['scale'] = keras_layer['config']['scale']
    # layer['weight_quantizer'] = BinaryQuantizer(bits=1)
    
    return layer, [shape for shape in input_shapes[0]]
