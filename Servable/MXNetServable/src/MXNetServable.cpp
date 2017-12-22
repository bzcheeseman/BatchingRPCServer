//
// Created by Aman LaChapelle on 11/5/17.
//
// BatchingRPCServer
// Copyright (c) 2017 Aman LaChapelle
// Full license at BatchingRPCServer/LICENSE.txt
//

/*
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "MXNetServable.hpp"

namespace Serving {

  MXNetServable::MXNetServable(
          const mx::Shape &input_shape,
          const mx::Shape &output_shape,
          const mx::DeviceType &type,
          const int &device_id
  ) : input_shape_(input_shape), output_shape_(output_shape), ctx_(type, device_id), bind_called_(false) {

    current_n_ = 0;
    current_batch_ = mx::NDArray(input_shape_, ctx_);
    args_map_["data"] = mx::NDArray(input_shape_, ctx_);

  }

  MXNetServable::~MXNetServable() {
    if (bind_called_) delete executor_;
  }

  ReturnCodes MXNetServable::AddToBatch(const TensorMessage &message, std::string client_id) {

    if (!bind_called_) {
      return ReturnCodes::NEED_BIND_CALL;
    }

    if (message.n() > input_shape_[0]) {
      return ReturnCodes::BATCH_TOO_LARGE;
    }

    if (message.k() != input_shape_[1] || message.nr() != input_shape_[2] || message.nc() != input_shape_[3]) {
      return ReturnCodes::SHAPE_INCORRECT;
    }

    if (message.n() + current_n_ > input_shape_[0]) { // early exit if we've got almost enough stuff here
      ready_to_process_ = true;
      ProcessCurrentBatch_();
      return ReturnCodes::NEXT_BATCH;
    }

    if (message.n() + current_n_ == input_shape_[0]) {
      ready_to_process_ = true;
    }

    for (int i = 0; i < message.n(); i++) {
      UpdateClientIDX(client_id, current_n_+i);
    }

    mx::NDArray message_array (message.buffer().data(),
                               mx::Shape(message.n(), input_shape_[1], input_shape_[2], input_shape_[3]),
                               ctx_);

    current_batch_.Slice(current_n_, current_n_+message.n()) = 0.f;
    current_batch_.Slice(current_n_, current_n_+message.n()) += message_array;
    current_n_ += message.n();

    if (ready_to_process_) {
      ProcessCurrentBatch_();
    }

    return ReturnCodes::OK;
  }

  TensorMessage MXNetServable::GetResult(std::string client_id) {

    mx::NDArray client_result = result_by_client_[client_id];
    std::vector<mx_uint> result_shape = client_result.GetShape();

    TensorMessage message;
    google::protobuf::RepeatedField<float> data(client_result.GetData(), client_result.GetData()+client_result.Size());
    message.mutable_buffer()->Swap(&data);

    message.set_n(result_shape[0]);
    message.set_k(result_shape[1]);
    message.set_nr(result_shape[2]);
    message.set_nc(result_shape[3]);
    message.set_client_id(client_id);

    return message;
  }

  void MXNetServable::Bind(mx::Symbol &net, std::map<std::string, mx::NDArray> &parameters) {
    servable_ = net;
    LoadParameters_(parameters);

    executor_ = servable_.SimpleBind(ctx_, args_map_,
                                     std::map<std::string, mx::NDArray>(), std::map<std::string, mx::OpReqType>(),
                                     aux_map_);

    bind_called_ = true;
  }

  void MXNetServable::Bind(const std::string &symbol_filename, const std::string &parameters_filename) {
    servable_ = mx::Symbol::Load(symbol_filename);
    servable_.InferArgsMap(ctx_, &args_map_, args_map_);
    std::map<std::string, mx::NDArray> parameters;
    mx::NDArray::Load(parameters_filename, nullptr, &parameters);
    LoadParameters_(parameters);

    executor_ = servable_.SimpleBind(ctx_, args_map_,
                                     std::map<std::string, mx::NDArray>(), std::map<std::string, mx::OpReqType>(),
                                     aux_map_);

    bind_called_ = true;
  }

  // Private methods //

  void MXNetServable::UpdateClientIDX(std::string &client_id, mx_uint &&msg_n) {
    idx_by_client_[client_id].push_back(msg_n);
  }

  void MXNetServable::LoadParameters_(std::map<std::string, mx::NDArray> &parameters) {
    for (const auto &k : parameters) {
      if (k.first.substr(0, 4) == "aux:") {
        auto name = k.first.substr(4, k.first.size() - 4);
        aux_map_[name] = k.second.Copy(ctx_);
      }
      if (k.first.substr(0, 4) == "arg:") {
        auto name = k.first.substr(4, k.first.size() - 4);
        args_map_[name] = k.second.Copy(ctx_);
      }
    }

    mx::NDArray::WaitAll();
  }

  void MXNetServable::ProcessCurrentBatch_() {

    current_batch_.CopyTo(&args_map_["data"]);

    executor_->Forward(false);

    mx::NDArray &result = executor_->outputs[0];
    mx::NDArray::WaitAll();

    for (auto &client_idx: idx_by_client_) {
      result_by_client_[client_idx.first] = mx::NDArray(mx::Shape(client_idx.second.size(), output_shape_[1]), ctx_);
      result_by_client_[client_idx.first] = 0.f;
      for (int i = 0; i < client_idx.second.size(); i++) {
        result_by_client_[client_idx.first].Slice(i, i+1) += result.Slice(client_idx.second[i], client_idx.second[i]+1);
      }
    }

    // Reset everyone
    current_batch_ = 0.f;
    ready_to_process_ = false;

  }

} // Serving


