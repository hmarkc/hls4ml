<p float="left">
   <img src="https://fastmachinelearning.github.io/hls4ml/img/logo.jpg" alt="hls4ml" width="400"/>
</p>

[![DOI](https://zenodo.org/badge/108329371.svg)](https://zenodo.org/badge/latestdoi/108329371)
[![PyPI version](https://badge.fury.io/py/hls4ml.svg)](https://badge.fury.io/py/hls4ml)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/hls4ml.svg)](https://pypi.org/project/hls4ml/)

A package for machine learning inference in FPGAs. We create firmware implementations of machine learning algorithms using high level synthesis language (HLS). We translate traditional open-source machine learning package models into HLS that can be configured for your use-case!

If you have any questions, comments, or ideas regarding hls4ml or just want to show us how you use hls4ml, don't hesitate to reach us through the [discussions](https://github.com/fastmachinelearning/hls4ml/discussions) tab.

# Documentation & Tutorial

For more information visit the webpage: [https://fastmachinelearning.org/hls4ml/](https://fastmachinelearning.org/hls4ml/)

Detailed tutorials on how to use `hls4ml`'s various functionalities can be found [here](https://github.com/hls-fpga-machine-learning/hls4ml-tutorial).

# Installation
```
pip install hls4ml
```

To install the extra dependencies for profiling: 

```
pip install hls4ml[profiling]
```

# Getting Started
### Creating an HLS project
```Python
import hls4ml

#Fetch a keras model from our example repository
#This will download our example model to your working directory and return an example configuration file
config = hls4ml.utils.fetch_example_model('KERAS_3layer.json')

print(config) #You can print the configuration to see some default parameters

#Convert it to a hls project
hls_model = hls4ml.converters.keras_to_hls(config)

# Print full list of example models if you want to explore more
hls4ml.utils.fetch_example_list()
```

### Building a project with Xilinx Vivado HLS (after downloading and installing from [here](https://www.xilinx.com/products/design-tools/vivado/integration/esl-design.html))
Note: Vitis HLS is not yet supported. Vivado HLS versions between 2018.2 and 2020.1 are recommended.

```Python
#Use Vivado HLS to synthesize the model
#This might take several minutes
hls_model.build()

#Print out the report if you want
hls4ml.report.read_vivado_report('my-hls-test')
```

# How to add custom layer
There are 2 ways to do this.

## Approach 1: Use the extension API of hls4ml
See https://fastmachinelearning.org/hls4ml/advanced/extension.html. 

## Approach 2: Modify the hls4ml codebase directly 
hls4ml is kind of like a compiler. It has a frontend for parsing and a backend for code generation. 
Suppose we want to add an **id** layer which outputs its input.
1. **hls4ml/utils/config.py/config_from_keras_model**:
Add the layer name in Keras to the supported layer list so that the parser recognizes our layer.
```Python
id_layers = ['ID']
#All supported layers
supported_layers = core_layers + dense_layers + conv_layers + pooling_layers + norm_layers + activation_layers + merge_layers + qkeras_layers + upsampling_layers + reshaping_layers + graph_layers + rnn_layers + id_layers + skip_layers
```
2. **hls4ml/converters/keras/**:
Add the keras handler for our **id** layer. Create a new file called id.py. The handler file will be automatically registered and used during parsing.
```Python
import math
from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler

@keras_handler('ID')
def parse_id_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('ID' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)
    return layer, [shape for shape in input_shapes[0]]
```
3. **hls4ml/model/layers.py**:
Add the IR object for our parsed layer. 
```Python
class ID(Layer):
    _expected_attributes = [
        Attribute('n_in')
    ]

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)
        self.set_attr('n_in', self.get_input_variable().size())
```
Add it to the layer_map.
```Python
layer_map = {
    'Input' : Input,
           ...
    'ID'    : ID,
}
```
4. **hls4ml/backends/vivado/pass/**:
Also add an **id_template.py** file to register our pass to the backend.
```Python
from hls4ml.backends.backend import get_backend
from hls4ml.model.layers import ID
from hls4ml.backends.template import LayerConfigTemplate, FunctionCallTemplate

# ID template

id_config_template = """
struct config{index} : nnet::id_config {{
    static const unsigned n_in = {n_in};
    static const unsigned io_type = nnet::{iotype};
}};\n"""

id_function_template = 'nnet::id<{input_t}, {output_t}, {config}>({input}, {output});'

id_include_list = ['nnet_utils/nnet_id.h', 'nnet_utils/nnet_id_stream.h']

class IDConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(ID)
        self.template = id_config_template

    def format(self, node):
        params = self._default_config_params(node)

        return self.template.format(**params)

class IDFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(ID, include_header=id_include_list)
        self.template = id_function_template

    def format(self, node):
        params = self._default_function_params(node)

        return self.template.format(**params)
```
5. **hls4ml/templates/vivado/nnet_utils/**:
Add 2 template files(1 for stream, 1 for parallel) for the generation the HLS code of our layer.

This is nnet_id.h.
```C++
#ifndef NNET_ID_H_
#define NNET_ID_H_

#include "ap_fixed.h"
#include "nnet_common.h"
#include <cmath>
#include <random>
#include <stdint.h>

namespace nnet {

struct id_config
{
    // IO size
    static const unsigned n_in = 10;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
};

// *************************************************
//       ID
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void id(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
  #pragma HLS PIPELINE
  for (int ii = 0; ii < CONFIG_T::n_in; ii++) { 
    res[ii] = data[ii];
  }
}
}

#endif

```
This is nnet_id_stream.h.
```C++
#ifndef NNET_ID_STREAM_H_
#define NNET_ID_STREAM_H_

#include <cmath>
#include "ap_fixed.h"
#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_types.h"
#include "nnet_stream.h"
#include "nnet_id.h"

namespace nnet {

// *************************************************
//       ID
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void id(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream) {

    typename data_T::value_type data[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=data complete

    typename res_T::value_type res[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=res complete

    DataPrepare: for(int i_in = 0; i_in < CONFIG_T::n_in / data_T::size; i_in++) {
        if (CONFIG_T::n_in / data_T::size > 1) {
            #pragma HLS PIPELINE
        }
        data_T data_pack = data_stream.read();
        DataPack: for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            #pragma HLS UNROLL
            data[i_in * data_T::size + i_pack] = data_pack[i_pack];
        }
    }

    ResWrite: for(unsigned i_out = 0; i_out < CONFIG_T::n_in / res_T::size; i_out++) {
        if (CONFIG_T::n_in / res_T::size > 1) {
            #pragma HLS PIPELINE
        }
        res_T res_pack;
        #pragma HLS DATA_PACK variable=res_pack
        ResPack: for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            #pragma HLS UNROLL
            res_pack[i_pack] = data[i_out * res_T::size + i_pack];
        }
        res_stream.write(res_pack);
    }
}
}

#endif
```

