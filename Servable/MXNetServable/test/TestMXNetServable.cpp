//
// Created by Aman LaChapelle on 12/9/17.
//
// BatchingRPCServer
// Copyright (c) 2017 Aman LaChapelle
// Full license at BatchingRPCServer/LICENSE.txt
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

#include "BatchingRPC.pb.h"
#include "Servable.hpp"
#include "MXNetServable.hpp"

#include "gtest/gtest.h"

namespace {
  namespace mx = mxnet::cpp;

  mx::Symbol SimpleSymbolFactory(int n_hidden) {
    mx::Symbol data = mx::Symbol::Variable("data");
    mx::Symbol m = mx::Symbol::Variable("m");
    mx::Symbol b = mx::Symbol::Variable("b");

    mx::Symbol result = mx::FullyConnected("fc1", data, m, b, n_hidden);

    return result;
  }

  Serving::TensorMessage ToMessage(mx::NDArray &arr) {
    std::vector<mx_uint> result_shape = arr.GetShape();

    Serving::TensorMessage message;
    google::protobuf::RepeatedField<float> data(arr.GetData(), arr.GetData()+arr.Size());
    message.mutable_buffer()->Swap(&data);

    message.set_n(result_shape[0]);
    message.set_k(result_shape[1]);
    message.set_nr(result_shape[2]);
    message.set_nc(result_shape[3]);
    message.set_client_id("data");

    return message;
  }

  void ThreadedAdd(Serving::Servable *servable, Serving::TensorMessage msg) {
    Serving::ReturnCodes r1 = servable->AddToBatch(msg);
    EXPECT_EQ(r1, Serving::ReturnCodes::OK);
  }

  class TestMXNetServable : public ::testing::Test {

  protected:
    void SetUp() override {

      ctx = new mx::Context(mx::kCPU, 0);

      fc = SimpleSymbolFactory(n_hidden);

      mx::NDArray m (mx::Shape(n_hidden, n_hidden), *ctx);
      m = 2.f;
      parms["arg:m"] = m;
      mx::NDArray b (mx::Shape(n_hidden), *ctx);
      b = 1.f;
      parms["arg:b"] = b;

      input = mx::NDArray(mx::Shape(1, 1, 1, n_hidden), *ctx);
      input = 1.f;

      zeros = mx::NDArray(mx::Shape(1, 1, 1, n_hidden), *ctx);
      zeros = 0.f;

      wrong_size = mx::NDArray(mx::Shape(1, 1, 1, n_hidden+1), *ctx);
      wrong_size = 1.f;

      too_big = mx::NDArray(mx::Shape(2, 1, 1, n_hidden), *ctx);
      too_big = 1.f;

      raw_args.net = fc;
      raw_args.parameters = parms;

      file_args.symbol_filename = "../../../Servable/MXNetServable/test/assets/squeezenet_v1.1-symbol.json";
      file_args.parameters_filename = "../../../Servable/MXNetServable/test/assets/squeezenet_v1.1-0000.params";

    }

    void TearDown() override {
      delete ctx;
    }

    mx::Context *ctx;
    int n_hidden = 2000;
    mx::Symbol fc;
    std::map<std::string, mx::NDArray> parms;
    mx::NDArray input;
    mx::NDArray zeros;
    mx::NDArray wrong_size;
    mx::NDArray too_big;

    Serving::RawBindArgs raw_args;
    Serving::FileBindArgs file_args;

  };

  TEST_F(TestMXNetServable, Bind) {
    Serving::MXNetServable servable (mx::Shape(1, 1, 1, n_hidden), mx::Shape(1, n_hidden), mx::kCPU, 1);

    EXPECT_NO_THROW(servable.Bind(raw_args));
  }

  TEST_F(TestMXNetServable, BindFile) {
    Serving::MXNetServable servable (mx::Shape(16, 3, 256, 256), mx::Shape(1, n_hidden), mx::kCPU, 1);

    EXPECT_NO_THROW(servable.Bind(file_args));
  }

  TEST_F(TestMXNetServable, Single) {
    Serving::MXNetServable servable (mx::Shape(1, 1, 1, n_hidden), mx::Shape(1, n_hidden), mx::kCPU, 1);

    servable.Bind(raw_args);

    Serving::TensorMessage msg = ToMessage(input);
    msg.set_client_id("test");

    Serving::ReturnCodes r = servable.AddToBatch(msg);
    EXPECT_EQ(r, Serving::ReturnCodes::OK);

    Serving::TensorMessage output = servable.GetResult("test");

    int buflen = msg.buffer().size();
    for (int i = 0; i < buflen; i++) {
      EXPECT_EQ(output.buffer(i), 2.f * n_hidden + 1);
    }

  }

  TEST_F(TestMXNetServable, NoBind) {
    Serving::MXNetServable servable (mx::Shape(1, 1, 1, n_hidden), mx::Shape(1, n_hidden), mx::kCPU, 1);

    Serving::TensorMessage msg = ToMessage(input);
    msg.set_client_id("no_bind");

    Serving::ReturnCodes r = servable.AddToBatch(msg);
    EXPECT_EQ(r, Serving::ReturnCodes::NEED_BIND_CALL);
  }

  TEST_F(TestMXNetServable, BadShape) {
    Serving::MXNetServable servable (mx::Shape(1, 1, 1, n_hidden), mx::Shape(1, n_hidden), mx::kCPU, 1);

    servable.Bind(raw_args);

    Serving::TensorMessage msg = ToMessage(wrong_size);
    msg.set_client_id("incorrect_shape");

    Serving::ReturnCodes r = servable.AddToBatch(msg);
    EXPECT_EQ(r, Serving::ReturnCodes::SHAPE_INCORRECT);
  }

  TEST_F(TestMXNetServable, TooBig) {
    Serving::MXNetServable servable (mx::Shape(1, 1, 1, n_hidden), mx::Shape(1, n_hidden), mx::kCPU, 1);

    servable.Bind(raw_args);

    Serving::TensorMessage msg = ToMessage(too_big);
    msg.set_client_id("too_big");

    Serving::ReturnCodes r = servable.AddToBatch(msg);
    EXPECT_EQ(r, Serving::ReturnCodes::BATCH_TOO_LARGE);
  }

  TEST_F(TestMXNetServable, NextBatch) {
    Serving::MXNetServable servable (mx::Shape(3, 1, 1, n_hidden), mx::Shape(1, n_hidden), mx::kCPU, 1);

    servable.Bind(raw_args);

    Serving::TensorMessage msg = ToMessage(too_big);
    msg.set_client_id("too_big");
    Serving::ReturnCodes r = servable.AddToBatch(msg);
    EXPECT_EQ(r, Serving::ReturnCodes::OK);

    msg = ToMessage(too_big);
    msg.set_client_id("too_big");

    r = servable.AddToBatch(msg);
    EXPECT_EQ(r, Serving::ReturnCodes::NEXT_BATCH);
  }

  TEST_F(TestMXNetServable, Multiple) {
    Serving::MXNetServable servable (mx::Shape(2, 1, 1, n_hidden), mx::Shape(1, n_hidden), mx::kCPU, 1);

    servable.Bind(raw_args);

    Serving::TensorMessage msg = ToMessage(input);
    msg.set_client_id("test");

    std::thread t1 (ThreadedAdd, &servable, msg);
    std::thread t2 (ThreadedAdd, &servable, msg);

    t1.join();
    t2.join();
    Serving::TensorMessage output = servable.GetResult("test");

    EXPECT_EQ(output.n(), 2);

    int buflen = msg.buffer().size();
    for (int i = 0; i < buflen; i++) {
      EXPECT_EQ(output.buffer(i), 2.f * n_hidden + 1);
    }

  }

  TEST_F(TestMXNetServable, MultipleClients) {
    Serving::MXNetServable servable (mx::Shape(3, 1, 1, n_hidden), mx::Shape(1, n_hidden), mx::kCPU, 1);

    servable.Bind(raw_args);

    Serving::TensorMessage msg = ToMessage(input);
    msg.set_client_id("test");
    Serving::TensorMessage z = ToMessage(zeros);
    z.set_client_id("zeros");

    std::thread t1 (ThreadedAdd, &servable, msg);
    std::thread t2 (ThreadedAdd, &servable, msg);
    std::thread tz (ThreadedAdd, &servable, z);

    Serving::TensorMessage output;
    int buflen;

    tz.join();
    output = servable.GetResult("zeros");
    EXPECT_EQ(output.n(), 1);
    buflen = msg.buffer().size();
    for (int i = 0; i < buflen; i++) {
      EXPECT_EQ(output.buffer(i), 1.f);
    }
    output.clear_buffer();

    t1.join();
    t2.join();
    output = servable.GetResult("test");
    EXPECT_EQ(output.n(), 2);
    buflen = msg.buffer().size();
    for (int i = 0; i < buflen; i++) {
      EXPECT_EQ(output.buffer(i), 2.f * n_hidden + 1);
    }

  }

  TEST_F(TestMXNetServable, UpdateBatchSuccess) {
    Serving::MXNetServable servable (mx::Shape(2, 1, 1, n_hidden), mx::Shape(1, n_hidden), mx::kCPU, 1);

    servable.Bind(raw_args);

    Serving::TensorMessage msg = ToMessage(input);
    msg.set_client_id("test");
    Serving::TensorMessage z = ToMessage(zeros);
    z.set_client_id("zeros");

    std::thread t1 (ThreadedAdd, &servable, msg);

    Serving::ReturnCodes r2 = servable.UpdateBatchSize(3);
    EXPECT_EQ(r2, Serving::ReturnCodes::OK);


    std::thread t2 (ThreadedAdd, &servable, msg);
    std::thread tz (ThreadedAdd, &servable, z);

    Serving::TensorMessage output;
    int buflen;

    tz.join();
    output = servable.GetResult("zeros");
    EXPECT_EQ(output.n(), 1);
    buflen = msg.buffer().size();
    for (int i = 0; i < buflen; i++) {
      EXPECT_EQ(output.buffer(i), 1.f);
    }
    output.clear_buffer();

    t1.join();
    t2.join();
    output = servable.GetResult("test");
    EXPECT_EQ(output.n(), 2);
    buflen = msg.buffer().size();
    for (int i = 0; i < buflen; i++) {
      EXPECT_EQ(output.buffer(i), 2.f * n_hidden + 1);
    }

  }

  TEST_F(TestMXNetServable, UpdateBatchFail) {
    Serving::MXNetServable servable (mx::Shape(3, 1, 1, n_hidden), mx::Shape(1, n_hidden), mx::kCPU, 1);

    servable.Bind(raw_args);

    Serving::TensorMessage msg = ToMessage(input);
    msg.set_client_id("test");
    Serving::TensorMessage z = ToMessage(zeros);
    z.set_client_id("zeros");

    std::thread t1 (ThreadedAdd, &servable, msg);
    std::thread t2 (ThreadedAdd, &servable, msg);

    Serving::ReturnCodes r3 = servable.UpdateBatchSize(1);
    EXPECT_EQ(r3, Serving::ReturnCodes::NEXT_BATCH);

    std::thread tz (ThreadedAdd, &servable, z);

    Serving::TensorMessage output;
    int buflen;

    tz.join();
    output = servable.GetResult("zeros");
    EXPECT_EQ(output.n(), 1);
    buflen = msg.buffer().size();
    for (int i = 0; i < buflen; i++) {
      EXPECT_EQ(output.buffer(i), 1.f);
    }
    output.clear_buffer();

    t1.join();
    t2.join();
    output = servable.GetResult("test");
    EXPECT_EQ(output.n(), 2);
    buflen = msg.buffer().size();
    for (int i = 0; i < buflen; i++) {
      EXPECT_EQ(output.buffer(i), 2.f * n_hidden + 1);
    }

  }

  // TODO: Add failure tests
}




