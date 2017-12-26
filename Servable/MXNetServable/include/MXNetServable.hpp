//
// Created by Aman LaChapelle on 11/5/17.
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


#ifndef BATCHING_RPC_SERVER_MXNETSERVABLE_HPP
#define BATCHING_RPC_SERVER_MXNETSERVABLE_HPP

// STL
#include <map>
#include <thread>

// MXNet
#include "mxnet-cpp/MxNetCpp.h"

// Project
#include "Servable.hpp"

// Generated
#include "BatchingRPC.pb.h"

namespace Serving {

  namespace mx = mxnet::cpp;

  struct RawBindArgs : public BindArgs {
    mx::Symbol net;
    std::map<std::string, mx::NDArray> parameters;
  };

  struct FileBindArgs : public BindArgs {
    std::string symbol_filename;
    std::string parameters_filename;
  };

  class MXNetServable: public Servable {
  public:
    MXNetServable(
            const mx::Shape &input_shape,
            const mx::Shape &output_shape,
            const mx::DeviceType &type,
            const int &device_id
    );

    ~MXNetServable() override ;

    ReturnCodes UpdateBatchSize(const int &new_size) override ;

    ReturnCodes AddToBatch(const TensorMessage &message) override ;

    ReturnCodes GetResult(const std::string &client_id, TensorMessage *message) override ;

    ReturnCodes Bind(BindArgs &args) override ;

  private:

    void BindExecutor_();
    void LoadParameters_(std::map<std::string, mx::NDArray> &parameters);
    void ProcessCurrentBatch_();

    // Basic I/O requirements
    std::atomic<bool> bind_called_;
    mx::Shape input_shape_;
    mx::Shape output_shape_;

    // Information for processing
    std::mutex input_mutex_;
    std::map<std::string, std::pair<mx_uint, mx_uint>> idx_by_client_;
    std::vector<mx::NDArray> current_batch_;

    mx_uint current_n_;

    std::mutex result_mutex_;
    std::condition_variable result_cv_;
    std::set<std::string> done_processing_by_client_;
    std::map<std::string, mx::NDArray> result_by_client_;

    // MXNet requirements for running
    mx::Context ctx_;
    mx::Symbol servable_; // see feature_extract.cpp for how to load models from files
    mx::Executor *executor_;
    std::map<std::string, mx::NDArray> args_map_; // inputs (data and model parameters) are args
    std::map<std::string, mx::NDArray> aux_map_; // everyone else is aux

  };

} // Serving


#endif //BATCHING_RPC_SERVER_MXNETSERVABLE_HPP
