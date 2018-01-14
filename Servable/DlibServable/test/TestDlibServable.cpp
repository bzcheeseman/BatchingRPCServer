//
// Created by Aman LaChapelle on 1/7/18.
//
// BatchingRPCServer
// Copyright (c) 2018 Aman LaChapelle
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

#include "BatchingRPC.pb.h"
#include "DlibServable.hpp"

#include "gtest/gtest.h"

namespace {

  using namespace dlib;

  using net_type = loss_multiclass_log<fc<
      10,
      relu<fc<
          84, relu<fc<
                  120, max_pool<
                           2, 2, 2, 2,
                           relu<con<
                               16, 5, 5, 1, 1,
                               max_pool<
                                   2, 2, 2, 2,
                                   relu<con<
                                       6, 5, 5, 1, 1,
                                       input<matrix<unsigned char>>>>>>>>>>>>>>;

  Serving::TensorMessage ToMessage(std::vector<matrix<unsigned char>> &&arr) {

    Serving::TensorMessage message;
    std::ostringstream buffer_stream (std::ios::binary);
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
    load_mnist_dataset(
        dirname, training_images, training_labels, testing_images,
        testing_labels);

    std::ifstream f("../../../Servable/DlibServable/test/assets/mnist_network.dat");
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
    serialize("../../../Servable/DlibServable/test/assets/mnist_network.dat") << net;
  }

  class TestDlibServable : public ::testing::Test {
  protected:
    void SetUp() override {
      TrainNetwork("../../../Servable/DlibServable/test/assets");
      deserialize("../../../Servable/DlibServable/test/assets/mnist_network.dat") >> raw_args.net;
      file_args.filename = "../../../Servable/DlibServable/test/assets/mnist_network.dat";

      std::vector<matrix<unsigned char>> training_images;
      std::vector<unsigned long> training_labels;
      load_mnist_dataset(
              "../../../Servable/DlibServable/test/assets", training_images, training_labels, input_,
              output_);

      std::cout << input_[0] << std::endl;

    }

    Serving::DlibRawBindArgs<net_type> raw_args;
    Serving::DlibFileBindArgs file_args;

    std::vector<matrix<unsigned char>> input_;
    std::vector<unsigned long> output_;
  };

  TEST_F(TestDlibServable, Bind) {
    Serving::DlibServable<net_type, matrix<unsigned char>, unsigned long>
        servable(4);

    EXPECT_NO_THROW(servable.Bind(raw_args));
  }

  TEST_F(TestDlibServable, BindFile) {
    Serving::DlibServable<net_type, matrix<unsigned char>, unsigned long>
            servable(4);

    EXPECT_NO_THROW(servable.Bind(file_args));
  }

  TEST_F(TestDlibServable, Single) {
    Serving::DlibServable<net_type, matrix<unsigned char>, unsigned long> servable(1);

    servable.Bind(raw_args);

    Serving::TensorMessage msg = ToMessage({input_[0]});
    msg.set_client_id("test");

    Serving::ReturnCodes r = servable.AddToBatch(msg);
    EXPECT_EQ(r, Serving::ReturnCodes::OK);

    Serving::TensorMessage output;
    r = servable.GetResult("test", &output);
    EXPECT_EQ(r, Serving::ReturnCodes::OK);
    std::istringstream output_buffer (output.serialized_buffer(), std::ios::binary);

    std::vector<unsigned long> results;
    deserialize(results, output_buffer);
    EXPECT_EQ(results[0], 7);
    // this particular image is a seven, since the net is trained we might as well run a prediction
  }

} // namespace
