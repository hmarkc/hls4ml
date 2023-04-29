
from hls4ml.backends.backend import get_backend
from hls4ml.model.layers import Masksembles
from hls4ml.backends.template import LayerConfigTemplate, FunctionCallTemplate

# Masksembles template

masksembles_config_template = """
struct config{index} : nnet::masksembles_config {{
    static const unsigned n_in = {n_in};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static const bool store_weights_in_bram = false;
    typedef {weight_t.name} weight_t;
}};\n"""

# isBayes must be set to True in config!
masksembles_function_template = 'nnet::masksembles<{input_t}, {output_t}, {config}>({input}, {output}, {w}, mask_index);'

masksembles_include_list = ['nnet_utils/nnet_masksembles.h', 'nnet_utils/nnet_masksembles_stream.h']

class MasksemblesConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Masksembles)
        self.template = masksembles_config_template

    def format(self, node):
        params = self._default_config_params(node)

        return self.template.format(**params)

class MasksemblesConfigTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Masksembles, include_header=masksembles_include_list)
        self.template = masksembles_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['w'] = node.get_weights('weight').name

        return self.template.format(**params)






