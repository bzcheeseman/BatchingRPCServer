//
// Created by Aman LaChapelle on 12/9/17.
//
// rpc_batch_scheduler
// Copyright (c) 2017 Aman LaChapelle
// Full license at rpc_batch_scheduler/LICENSE.txt
//

/*
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
 */

#include <map>

#include "mxnet-cpp/MxNetCpp.h"

#include "TensorMessage.pb.h"
#include "Servable.hpp"
#include "MXNetServable.hpp"

#include "gtest/gtest.h"

namespace {
  namespace mx = mxnet::cpp;

  mx::Symbol SimpleSymbolFactory(int n_hidden) {
    mx::Symbol data = mx::Symbol::Variable("data");
    mx::Symbol m = mx::Symbol::Variable("m");
    mx::Symbol b = mx::Symbol::Variable("b");

    mx::Symbol result = mx::FullyConnected(data, m, b, n_hidden);

    return result;
  }

  Serving::TensorMessage ToMessage(mx::NDArray &arr) {
    std::vector<mx_uint> result_shape = arr.GetShape();

    std::cout << arr.Size() << std::endl;
    std::cout << arr << std::endl;

    Serving::TensorMessage message;
    google::protobuf::RepeatedField<float> data(arr.GetData(), arr.GetData()+arr.Size());
    message.mutable_buffer()->Swap(&data);

    message.set_n(result_shape[0]);
    message.set_k(result_shape[1]);
    message.set_nr(result_shape[2]);
    message.set_nc(result_shape[3]);
    message.set_name("sample_input");

    return message;
  }

  class TestMXNetServable : public ::testing::Test {

  protected:
    void SetUp() override {

      ctx = new mx::Context(mx::kCPU, 0);

      fc = SimpleSymbolFactory(n_hidden);

      mx::NDArray m (mx::Shape(1, 1, n_hidden, n_hidden), *ctx);
      m = 2.f;
      parms["arg:m"] = m;
      mx::NDArray b (mx::Shape(1, 1, 1, n_hidden), *ctx);
      b = 1.f;
      parms["arg:b"] = b;

      m = 1.f;
      parms_noop["arg:m"] = m;
      b = 0.f;
      parms_noop["arg:b"] = b;

      input = mx::NDArray(mx::Shape(1, 1, 1, n_hidden), *ctx);
      input = 1.f;
      std::cout << input << std::endl;

    }

    void TearDown() override {
      delete ctx;
    }

    mx::Context *ctx;
    int n_hidden = 5;
    mx::Symbol fc;
    std::map<std::string, mx::NDArray> parms;
    std::map<std::string, mx::NDArray> parms_noop;
    mx::NDArray input;

  };

  TEST_F(TestMXNetServable, Single) {
    Serving::internal::MXNetServable servable (mx::Shape(1, 1, 1, n_hidden), mx::Shape(1, 1, 1, n_hidden), mx::kCPU, 1);

    servable.Bind(fc, parms_noop);

    Serving::TensorMessage msg = ToMessage(input);

    servable.AddToBatch(msg, "test");

    Serving::TensorMessage output = servable.GetResult("test");
//    std::string test_result = "test_result";
//    EXPECT_STREQ(output.name(), test_result);

    int buflen = msg.buffer().size();
    for (int i = 0; i < buflen; i++) {
      EXPECT_EQ(output.buffer(i), msg.buffer(i));
    }

  }
}

// TODO: write the rest of the tests




