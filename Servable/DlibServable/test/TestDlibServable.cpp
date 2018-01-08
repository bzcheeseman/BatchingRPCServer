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
    }

    Serving::DlibRawBindArgs<net_type> raw_args;
    Serving::DlibFileBindArgs file_args;
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

} // namespace
