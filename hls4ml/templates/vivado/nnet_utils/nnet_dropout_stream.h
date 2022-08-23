//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_DROPOUT_STREAM_H_
#define NNET_DROPOUT_STREAM_H_

#include <cmath>
#include "ap_fixed.h"
#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_types.h"
#include "nnet_stream.h"
#include "nnet_dropout.h"

namespace nnet {

// *************************************************
//       Bayesian Dropout
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void dropout(hls::stream<data_T> &data, hls::stream<res_T> &res, int seed) {
    #pragma HLS pipeline
    
  static std::minstd_rand generator(seed);
  typename data_T::value_type keep_rate = 1 - CONFIG_T::drop_rate;
  typename data_T::value_type max = generator.max(); 
  bool random_array[CONFIG_T::n_in];
    RandomNumberLoop: for (int i = 0; i < CONFIG_T::n_in; i++) {
        random_array[i] =
            ((typename data_T::value_type)generator() / max) < keep_rate;
    }
    DropoutLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
        #pragma HLS DATA_PACK variable=out_data

        DropoutPackLoop: for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            typename data_T::value_type zero = {};
            typename data_T::value_type temp =
                random_array[i * res_T::size + j] ? in_data[j] : zero;
            out_data[j] = temp * keep_rate;
        }

        res.write(out_data);
    }
}
}

#endif