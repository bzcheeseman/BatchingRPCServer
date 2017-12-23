//
// Created by Aman LaChapelle on 12/22/17.
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

#include "Servable.hpp"
#include "MXNetServable.hpp"

#include "TBServer.hpp"

#include "gtest/gtest.h"

#include "BatchingRPC.pb.h"
#include "BatchingRPC.grpc.pb.h"

namespace Serving {
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
      google::protobuf::RepeatedField<float> data(arr.GetData(), arr.GetData() + arr.Size());
      message.mutable_buffer()->Swap(&data);

      message.set_n(result_shape[0]);
      message.set_k(result_shape[1]);
      message.set_nr(result_shape[2]);
      message.set_nc(result_shape[3]);
      message.set_client_id("data");

      return message;
    }

    class TestIntegration : public ::testing::Test {
    protected:
      void SetUp() override {

        ctx = new mx::Context(mx::kCPU, 0);

        raw_args.net = SimpleSymbolFactory(n_hidden);

        mx::NDArray m(mx::Shape(n_hidden, n_hidden), *ctx);
        m = 2.f;
        raw_args.parameters["arg:m"] = m;
        mx::NDArray b(mx::Shape(n_hidden), *ctx);
        b = 1.f;
        raw_args.parameters["arg:b"] = b;

        input = mx::NDArray(mx::Shape(1, 1, 1, n_hidden), *ctx);
        input = 1.f;

        servable = new MXNetServable(mx::Shape(1, 1, 1, n_hidden), mx::Shape(1, n_hidden), mx::kCPU, 1);
        servable->Bind(raw_args);
        srv = new TBServer(servable);
        srv->StartInsecure("localhost:50051");

        msg = ToMessage(input);

      }

      void TearDown() override {
        srv->Stop();
        delete srv;
        delete servable;
      }

      Serving::Servable *servable;
      Serving::TBServer *srv;

      Serving::TensorMessage msg;

      mx::Context *ctx;
      int n_hidden = 600;
      mx::NDArray input;

      Serving::RawBindArgs raw_args;
    };

    TEST_F(TestIntegration, Connect) {
      std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel("localhost:50051",
                                                                   grpc::InsecureChannelCredentials());
      std::unique_ptr<BatchingServable::Stub> stub = BatchingServable::NewStub(channel);

      grpc::ClientContext context;

      ConnectionReply rep;

      grpc::Status status = stub->Connect(&context, ConnectionRequest(), &rep);

      EXPECT_TRUE(status.ok());
      EXPECT_FALSE(rep.client_id().empty());
    }

    TEST_F(TestIntegration, SetBatchSize) {
      std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel("localhost:50051",
                                                                   grpc::InsecureChannelCredentials());
      std::unique_ptr<BatchingServable::Stub> stub = BatchingServable::NewStub(channel);

      grpc::ClientContext context;

      AdminRequest req;
      req.set_new_batch_size(5);
      AdminReply rep;

      grpc::Status status = stub->SetBatchSize(&context, req, &rep);

      EXPECT_TRUE(status.ok());
    }

    TEST_F(TestIntegration, Process) {
      std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel("localhost:50051",
                                                                   grpc::InsecureChannelCredentials());
      std::unique_ptr<BatchingServable::Stub> stub = BatchingServable::NewStub(channel);

      ConnectionReply rep;
      grpc::Status status;

      {
        grpc::ClientContext context;
        status = stub->Connect(&context, ConnectionRequest(), &rep);
        EXPECT_TRUE(status.ok());
        EXPECT_FALSE(rep.client_id().empty());
      }

      msg.set_client_id(rep.client_id());

      TensorMessage tensor_reply;

      {
        grpc::ClientContext context;
        status = stub->Process(&context, msg, &tensor_reply);
        EXPECT_TRUE(status.ok());

        int buflen = tensor_reply.buffer().size();
        for (int i = 0; i < buflen; i++) {
          EXPECT_EQ(tensor_reply.buffer(i), 2.f * n_hidden + 1);
        }
      }

      //
    }
  }
} // Serving


