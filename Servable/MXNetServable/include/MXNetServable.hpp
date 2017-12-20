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


#ifndef BATCHING_RPC_SERVER_TENSORBATCHINGSERVABLE_HPP
#define BATCHING_RPC_SERVER_TENSORBATCHINGSERVABLE_HPP

#include <map>

#include "mxnet-cpp/MxNetCpp.h"

#include "BatchingRPC.pb.h"
#include "Servable.hpp"

namespace Serving {

  namespace mx = mxnet::cpp;

  class MXNetServable: public Servable {
  public:
    MXNetServable(
            const mx::Shape &input_shape,
            const mx::Shape &output_shape,
            const mx::DeviceType &type,
            const int &device_id
    );

    ~MXNetServable();

    ReturnCodes AddToBatch(TensorMessage &message, std::string client_id) override;

    TensorMessage GetResult(std::string client_id) override;

    void Bind(mx::Symbol &net, std::map<std::string, mx::NDArray> &parameters);
    void Bind(const std::string &symbol_filename, const std::string &parameters_filename);

  private:

    void UpdateClientIDX(std::string &client_id, mx_uint &&msg_n);
    void LoadParameters_(std::map<std::string, mx::NDArray> &parameters);
    void ProcessCurrentBatch_();

    // Basic I/O requirements
    bool bind_called_;
    mx::Shape input_shape_;
    mx::Shape output_shape_;

    // Information for processing
    std::map<std::string, std::vector<mx_uint>> idx_by_client_;
    mx_uint current_n_;
    mx::NDArray current_batch_;
    std::map<std::string, mx::NDArray> result_by_client_;

    // MXNet requirements for running
    mx::Context ctx_;
    mx::Symbol servable_; // see feature_extract.cpp for how to load models from files
    mx::Executor *executor_;
    std::map<std::string, mx::NDArray> args_map_; // inputs (data and model parameters) are args
    std::map<std::string, mx::NDArray> aux_map_; // everyone else is aux

    // Serving vars
    bool ready_to_process_ = false;
  };

} // Serving


#endif //BATCHING_RPC_SERVER_TENSORBATCHINGSERVABLE_HPP