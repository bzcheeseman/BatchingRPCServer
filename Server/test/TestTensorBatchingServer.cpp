//
// Created by Aman LaChapelle on 12/20/17.
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

#include "../include/TensorBatchingServer.hpp"
#include "Servable.hpp"

#include <grpc++/grpc++.h>
#include <thread>

#include "gtest/gtest.h"

namespace Serving {
  namespace {

    class MockedServable : public Servable {
    public:
      MockedServable() = default;
      ~MockedServable() override = default;

      ReturnCodes AddToBatch(const TensorMessage &message, std::string client_id) override {
        msg = message;
        return OK;
      }

      TensorMessage GetResult(std::string client_id) override {
        return msg;
      }

    private:
      TensorMessage msg;
    };

    class TestTensorBatchingServer : public ::testing::Test {
    protected:
      void SetUp() override {

        servable = new MockedServable ();
        srv = new TBServer (servable);
        srv->Start("localhost:50051");

        lim = 10000;
        for (int i = 0; i < lim; i++) {
          msg.mutable_buffer()->Add((float)i);
        }
        msg.set_n(lim);

      }

      void TearDown() override {
        srv->Stop();
        delete srv;
        delete servable;
      }

      int lim;
      Serving::Servable *servable;
      Serving::TBServer *srv;

      TensorMessage msg;
    };

    TEST_F(TestTensorBatchingServer, Connect) {
      std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials());
      std::unique_ptr<BatchingServable::Stub> stub = BatchingServable::NewStub(channel);

      grpc::ClientContext context;

      ConnectionReply rep;

      grpc::Status status = stub->Connect(&context, ConnectionRequest(), &rep);

      EXPECT_TRUE(status.ok());
      EXPECT_FALSE(rep.client_id().empty());

      std::cout << "\nClient ID: " << rep.client_id() << std::endl;
    }

    TEST_F(TestTensorBatchingServer, Process) {
      std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials());
      std::unique_ptr<BatchingServable::Stub> stub = BatchingServable::NewStub(channel);

      grpc::ClientContext context;

      ConnectionReply rep;

      grpc::Status status = stub->Connect(&context, ConnectionRequest(), &rep);

      EXPECT_TRUE(status.ok());
      EXPECT_FALSE(rep.client_id().empty());

      msg.set_client_id(rep.client_id());
      std::cout << msg.client_id() << std::endl;

      TensorMessage tensor_reply;
      status = stub->Process(&context, msg, &tensor_reply);

      EXPECT_TRUE(status.ok());

      for (int i = 0; i < lim; i++) {
        EXPECT_EQ(tensor_reply.buffer(i), (float)i);
      }
    }

  }
} // Serving

