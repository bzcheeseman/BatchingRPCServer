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

#include <dlib/data_io.h>
#include <sstream>

#include "DlibServable.hpp"
#include "Servable.hpp"

#include "TBServer.hpp"

#include "gtest/gtest.h"

#include "BatchingRPC.grpc.pb.h"
#include "BatchingRPC.pb.h"

#include <future>

namespace Serving {
namespace {

using namespace dlib;

using net_type = loss_multiclass_log<fc<
    10,
    relu<fc<
        84,
        relu<fc<
            120,
            max_pool<2, 2, 2, 2,
                     relu<con<16, 5, 5, 1, 1,
                              max_pool<2, 2, 2, 2,
                                       relu<con<6, 5, 5, 1, 1,
                                                input<matrix<
                                                    unsigned char>>>>>>>>>>>>>>;

Serving::TensorMessage ToMessage(std::vector<matrix<unsigned char>> &&arr) {

  Serving::TensorMessage message;
  std::ostringstream buffer_stream(std::ios::binary);
  serialize(arr, buffer_stream);
  message.set_serialized_buffer(buffer_stream.str());
  message.set_n(arr.size());

  return message;
}

void TrainNetwork(const std::string &dirname) {
  std::vector<matrix<unsigned char>> training_images;
  std::vector<unsigned long> training_labels;
  std::vector<matrix<unsigned char>> testing_images;
  std::vector<unsigned long> testing_labels;
  load_mnist_dataset(dirname, training_images, training_labels, testing_images,
                     testing_labels);

  std::ifstream f(
      "../Servable/DlibServable/test/assets/mnist_network.dat");
  if (f.is_open()) {
    f.close();
    return;
  }

  net_type net;
  dnn_trainer<net_type> trainer(net);
  trainer.set_learning_rate(0.01);
  trainer.set_min_learning_rate(0.0001);
  trainer.set_mini_batch_size(128);
  trainer.be_verbose();
  trainer.set_synchronization_file("mnist_sync", std::chrono::seconds(20));
  trainer.train(training_images, training_labels);
  net.clean();
  serialize("../Servable/DlibServable/test/assets/mnist_network.dat")
      << net;
}

void ThreadProcess(std::unique_ptr<BatchingServer::Stub> &stub,
                   TensorMessage msg, TensorMessage &reply) {
  grpc::Status status;
  ConnectionReply rep;

  {
    grpc::ClientContext context;
    stub->Connect(&context, ConnectionRequest(), &rep);
  }

  msg.set_client_id(rep.client_id());

  {
    grpc::ClientContext context;
    stub->Process(&context, msg, &reply);
  }
}

class TestIntegrationDlib : public ::testing::Test {
protected:
  void SetUp() override {
    TrainNetwork("../Servable/DlibServable/test/assets");
    deserialize(
        "../Servable/DlibServable/test/assets/mnist_network.dat") >>
        raw_args.net;
    file_args.filename =
        "../Servable/DlibServable/test/assets/mnist_network.dat";

    std::vector<matrix<unsigned char>> training_images;
    std::vector<unsigned long> training_labels;
    load_mnist_dataset("../Servable/DlibServable/test/assets",
                       training_images, training_labels, input_, output_);

    Serving::DlibServable<net_type, matrix<unsigned char>, unsigned long> *servable =
            new Serving::DlibServable<net_type, matrix<unsigned char>, unsigned long>(1);
    servable->Bind(raw_args);
    srv = new TBServer(servable);

    srv->StartInsecure("localhost:50051");

    msg = ToMessage({input_[0]});
  }

  Serving::TBServer *srv;
  Serving::DlibRawBindArgs<net_type> raw_args;
  Serving::DlibFileBindArgs file_args;

  std::vector<matrix<unsigned char>> input_;
  std::vector<unsigned long> output_;
  Serving::TensorMessage msg;
};

TEST_F(TestIntegrationDlib, Connect) {
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials());
  std::unique_ptr<BatchingServer::Stub> stub = BatchingServer::NewStub(channel);

  grpc::ClientContext context;

  ConnectionReply rep;

  grpc::Status status = stub->Connect(&context, ConnectionRequest(), &rep);

  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(rep.client_id().empty());
}

TEST_F(TestIntegrationDlib, SetBatchSize) {
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials());
  std::unique_ptr<BatchingServer::Stub> stub = BatchingServer::NewStub(channel);

  grpc::ClientContext context;

  AdminRequest req;
  req.set_new_batch_size(5);
  AdminReply rep;

  grpc::Status status = stub->SetBatchSize(&context, req, &rep);

  EXPECT_TRUE(status.ok());
}

TEST_F(TestIntegrationDlib, Process) {
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials());
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

    std::istringstream output_buffer(tensor_reply.serialized_buffer(),
                                     std::ios::binary);

    std::vector<unsigned long> results;
    deserialize(results, output_buffer);
    // this particular image is a seven, since the net is trained we might as well
    // run a prediction
    EXPECT_EQ(results[0], 7);
  }
}

TEST_F(TestIntegrationDlib, ThreadedProcessSingle) {
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials());
  std::unique_ptr<BatchingServer::Stub> stub = BatchingServer::NewStub(channel);

  TensorMessage tensor_reply;

  std::thread processing_thread(ThreadProcess, std::ref(stub), msg,
                                std::ref(tensor_reply));

  processing_thread.join();

  std::istringstream output_buffer(tensor_reply.serialized_buffer(),
                                   std::ios::binary);
  std::vector<unsigned long> results;
  deserialize(results, output_buffer);
  // this particular image is a seven, since the net is trained we might as well
  // run a prediction
  EXPECT_EQ(results[0], 7);
}

TEST_F(TestIntegrationDlib, ThreadedProcessMultiple_SingleBatch) {
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials());
  std::unique_ptr<BatchingServer::Stub> stub = BatchingServer::NewStub(channel);

  int batch_size = 50;

  {
    grpc::ClientContext context;
    AdminRequest req;
    req.set_new_batch_size(batch_size);
    AdminReply rep;

    grpc::Status status = stub->SetBatchSize(&context, req, &rep);
    EXPECT_TRUE(status.ok());
  }

  std::vector<std::thread> request_threads;
  std::vector<TensorMessage> results(batch_size);

  for (int i = 0; i < batch_size; i++) {
    request_threads.emplace_back(
        std::thread(ThreadProcess, std::ref(stub), msg, std::ref(results[i])));
  }

  for (int i = 0; i < batch_size; i++) {
    request_threads[i].join();
    TensorMessage &tensor_reply = results[i];

    std::istringstream output_buffer(tensor_reply.serialized_buffer(),
                                     std::ios::binary);
    std::vector<unsigned long> results;
    deserialize(results, output_buffer);
    // this particular image is a seven, since the net is trained we might as well
    // run a prediction
    EXPECT_EQ(results[0], 7);
    tensor_reply.clear_buffer();
  }
}

TEST_F(TestIntegrationDlib, ThreadedProcessMultiple_MultiBatch) {
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials());
  std::unique_ptr<BatchingServer::Stub> stub = BatchingServer::NewStub(channel);

  int batch_size = 50;

  {
    grpc::ClientContext context;
    AdminRequest req;
    req.set_new_batch_size(10);
    AdminReply rep;

    grpc::Status status = stub->SetBatchSize(&context, req, &rep);
    EXPECT_TRUE(status.ok());
  }

  std::vector<std::thread> request_threads;
  std::vector<TensorMessage> results(batch_size);

  for (int i = 0; i < batch_size; i++) {
    request_threads.emplace_back(
        std::thread(ThreadProcess, std::ref(stub), msg, std::ref(results[i])));
  }

  for (int i = 0; i < batch_size; i++) {
    request_threads[i].join();
    TensorMessage &tensor_reply = results[i];

    std::istringstream output_buffer(tensor_reply.serialized_buffer(),
                                     std::ios::binary);
    std::vector<unsigned long> results;
    deserialize(results, output_buffer);
    // this particular image is a seven, since the net is trained we might as well
    // run a prediction
    EXPECT_EQ(results[0], 7);
    tensor_reply.clear_buffer();
  }
}
}
} // namespace Serving::
