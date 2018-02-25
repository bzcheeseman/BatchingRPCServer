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

#include "Servable.hpp"
#include "TBServer.hpp"

#include <grpc++/grpc++.h>

#include <fstream>

#include "gtest/gtest.h"

namespace Serving {
namespace {

class EchoServable : public Servable {
public:
  EchoServable() = default;
  ~EchoServable() override = default;

  ReturnCodes SetBatchSize(const int &new_size) override { return OK; }

  ReturnCodes AddToBatch(const TensorMessage &message) override {
    msg = message;
    return OK;
  }

  ReturnCodes GetResult(const std::string &client_id,
                        TensorMessage *message) override {
    *message = msg;
    return OK;
  }

  ReturnCodes Bind(BindArgs &args) override { return OK; }

private:
  TensorMessage msg;
};

class TestTBServer : public ::testing::Test {
protected:
  void SetUp() override {

    EchoServable *servable = new EchoServable();
    srv = new TBServer(servable);
    srv->StartSSL("localhost:50051", "server-key.pem", "server-cert.pem");

    std::ifstream in_file;
    std::string cert, tmp;
    in_file.open("server-cert.pem");
    while (in_file.good()) {
      std::getline(in_file, tmp);
      cert += tmp + "\n";
    }
    in_file.close();

    client_creds = {cert, ""};

    lim = 100000;
    for (int i = 0; i < lim; i++) {
      msg.mutable_buffer()->Add((float)i);
    }
    msg.set_n(lim);
    msg.set_k(0);
    msg.set_nr(0);
    msg.set_nc(0);
  }

  void TearDown() override {
    srv->Stop();
    delete srv;
  }

  int lim;
  Serving::TBServer *srv;
  grpc::SslCredentialsOptions client_creds;

  TensorMessage msg;
};

TEST_F(TestTBServer, Connect) {
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      "localhost:50051", grpc::SslCredentials(client_creds));
  std::unique_ptr<BatchingServer::Stub> stub = BatchingServer::NewStub(channel);

  grpc::ClientContext context;

  ConnectionReply rep;

  grpc::Status status = stub->Connect(&context, ConnectionRequest(), &rep);

  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(rep.client_id().empty());
}

TEST_F(TestTBServer, SetBatchSize) {
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      "localhost:50051", grpc::SslCredentials(client_creds));
  std::unique_ptr<BatchingServer::Stub> stub = BatchingServer::NewStub(channel);

  grpc::ClientContext context;

  AdminRequest req;
  req.set_new_batch_size(5);
  AdminReply rep;

  grpc::Status status = stub->SetBatchSize(&context, req, &rep);

  EXPECT_TRUE(status.ok());
}

TEST_F(TestTBServer, Process) {
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      "localhost:50051", grpc::SslCredentials(client_creds));
  std::unique_ptr<BatchingServer::Stub> stub = BatchingServer::NewStub(channel);

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
    EXPECT_EQ(tensor_reply.n(), lim);
  }

  for (int i = 0; i < lim; i++) {
    EXPECT_EQ(tensor_reply.buffer(i), (float)i);
  }
}

TEST_F(TestTBServer, FailProcess) {
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      "localhost:50051", grpc::SslCredentials(client_creds));
  std::unique_ptr<BatchingServer::Stub> stub = BatchingServer::NewStub(channel);

  msg.set_client_id("test");

  TensorMessage tensor_reply;
  grpc::Status status;

  {
    grpc::ClientContext context;
    status = stub->Process(&context, msg, &tensor_reply);
    EXPECT_FALSE(status.ok());
    EXPECT_TRUE(status.error_code() == grpc::FAILED_PRECONDITION);
  }
}

TEST_F(TestTBServer, Reconnect) {
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      "localhost:50051", grpc::SslCredentials(client_creds));
  std::unique_ptr<BatchingServer::Stub> stub = BatchingServer::NewStub(channel);

  ConnectionReply rep;
  grpc::Status status;

  {
    grpc::ClientContext context;
    status = stub->Connect(&context, ConnectionRequest(), &rep);
    EXPECT_TRUE(status.ok());
    EXPECT_FALSE(rep.client_id().empty());
  }

  std::string first_id = rep.client_id();

  {
    grpc::ClientContext context;
    status = stub->Connect(&context, ConnectionRequest(), &rep);
    EXPECT_TRUE(status.ok());
    EXPECT_FALSE(rep.client_id().empty());
  }

  EXPECT_STRNE(first_id.c_str(), rep.client_id().c_str());
}

TEST_F(TestTBServer, MultipleProcess) {
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      "localhost:50051", grpc::SslCredentials(client_creds));
  std::unique_ptr<BatchingServer::Stub> stub = BatchingServer::NewStub(channel);

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
    EXPECT_EQ(tensor_reply.n(), lim);
  }

  for (int i = 0; i < lim; i++) {
    EXPECT_EQ(tensor_reply.buffer(i), (float)i);
  }

  {
    grpc::ClientContext context;
    status = stub->Process(&context, msg, &tensor_reply);
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(tensor_reply.n(), lim);
  }

  for (int i = 0; i < lim; i++) {
    EXPECT_EQ(tensor_reply.buffer(i), (float)i);
  }
}
}
} // namespace Serving::