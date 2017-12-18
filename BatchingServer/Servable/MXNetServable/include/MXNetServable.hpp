//
// Created by Aman LaChapelle on 11/5/17.
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


#ifndef RPC_BATCH_SCHEDULER_TENSORBATCHINGSERVABLE_HPP
#define RPC_BATCH_SCHEDULER_TENSORBATCHINGSERVABLE_HPP

#include <map>

#include "mxnet-cpp/MxNetCpp.h"

#include "BatchingRPC.pb.h"
#include "Servable.hpp"

namespace Serving { namespace internal {

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

    void LoadParameters_(std::map<std::string, mx::NDArray> &parameters);
    void ProcessCurrentBatch_();

    // Basic I/O requirements
    bool bind_called_;
    mx::Shape input_shape_;
    mx::Shape output_shape_;

    // Information for processing
    std::map<std::string, std::pair<mx_uint, mx_uint>> idx_by_client_;
    mx_uint current_n_;
    mx::NDArray current_batch_;
    std::map<std::string, mx::NDArray> result_by_client_;

    // MXNet requirements for running
    mx::Context ctx_;
    mx::Symbol servable_; // see feature_extract.cpp for how to load models from files
    mx::Executor *executor_;
    std::map<std::string, mx::NDArray> args_map_;
    std::map<std::string, mx::NDArray> aux_map_;

    // Serving vars
    bool ready_to_process_ = false;
  };

}} // Serving::internal


#endif //RPC_BATCH_SCHEDULER_TENSORBATCHINGSERVABLE_HPP
