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
  ) : input_shape_(input_shape), output_shape_(output_shape), ctx_(type, device_id),
      bind_called_(false), current_n_(0) {

    args_map_["data"] = mx::NDArray(input_shape_, ctx_);

  }

  MXNetServable::~MXNetServable() {
    if (bind_called_) delete executor_;
  }

  ReturnCodes MXNetServable::UpdateBatchSize(const int &new_size) {
    std::lock_guard<std::mutex> guard_input (input_mutex_);

    if (new_size <= current_n_) {
      return ReturnCodes::NEXT_BATCH;
    }

    // Reshape the input
    input_shape_ = mx::Shape(new_size, input_shape_[1], input_shape_[2], input_shape_[3]);

    // Re-bind the executor with the new batch size
    args_map_["data"] = mx::NDArray(input_shape_, ctx_);
    BindExecutor_();

    return ReturnCodes::OK;

  }

  // similar to enqueue in https://github.com/Youka/ThreadPool/blob/master/ThreadPool.hpp
  ReturnCodes MXNetServable::AddToBatch(const TensorMessage &message) {

    const std::string &client_id = message.client_id();
    std::vector<float> message_data (message.buffer().data(), message.buffer().data() + message.buffer_size());

    if (!bind_called_) {
      return ReturnCodes::NEED_BIND_CALL;
    }

    if (message.n() > input_shape_[0]) {
      return ReturnCodes::BATCH_TOO_LARGE;
    }

    if (message.k() != input_shape_[1] || message.nr() != input_shape_[2] || message.nc() != input_shape_[3]) {
      return ReturnCodes::SHAPE_INCORRECT;
    }

    {

      std::lock_guard<std::mutex> guard_input(input_mutex_);

      if (message.n() + current_n_ > input_shape_[0]) {
        return ReturnCodes::NEXT_BATCH;
      }

      result_by_client_.erase(client_id); // clears room for the new result

      if (idx_by_client_.find(client_id) == idx_by_client_.end()) {
        idx_by_client_[client_id] = std::make_pair(current_n_, current_n_ + message.n());
      } else {
        idx_by_client_[client_id].second += message.n();
      }

      current_batch_.emplace_back(
              mx::NDArray(message.buffer().data(),
                          mx::Shape(message.n(), input_shape_[1], input_shape_[2], input_shape_[3]), ctx_)
      );

      current_n_ += message.n();
      if (current_n_ <= input_shape_[0]) {
        if (current_n_ == input_shape_[0]) {
          ProcessCurrentBatch_();
        }
      }

    }

    return ReturnCodes::OK;
  }

  ReturnCodes MXNetServable::GetResult(const std::string &client_id, TensorMessage *message) {

    std::unique_lock<std::mutex> lk(result_mutex_);
    result_cv_.wait(lk, [&, this](){ // This will block forever if the client only calls GetResult and never AddToBatch
      auto done = done_processing_by_client_.find(client_id);
      if (done != done_processing_by_client_.end()) {
        done = done_processing_by_client_.erase(done);
        return true;
      }
      else {
        return false;
      }
    });

    auto result = result_by_client_.find(client_id);
    mx::NDArray &result_array = result->second;

    std::vector<mx_uint> result_shape = result_array.GetShape();

    google::protobuf::RepeatedField<float> data(result_array.GetData(), result_array.GetData()+result_array.Size());
    message->mutable_buffer()->Swap(&data);
    result = result_by_client_.erase(result);


    message->set_n(result_shape[0]);
    message->set_k(result_shape[1]);
    message->set_nr(result_shape[2]);
    message->set_nc(result_shape[3]);
    message->set_client_id(client_id);

    return ReturnCodes::OK;

  }

  ReturnCodes MXNetServable::Bind(BindArgs &args) {
    bind_called_ = true;

    try {
      RawBindArgs &raw_args = dynamic_cast<RawBindArgs &>(args);
      servable_ = raw_args.net;
      LoadParameters_(raw_args.parameters);
      BindExecutor_();
      return ReturnCodes::OK;
    }
    catch (std::bad_cast &e) {
      ;
    }

    try {
      FileBindArgs &file_args = dynamic_cast<FileBindArgs &>(args);
      servable_ = mx::Symbol::Load(file_args.symbol_filename);
      std::map<std::string, mx::NDArray> parameters = mx::NDArray::LoadToMap(file_args.parameters_filename);
      LoadParameters_(parameters);
      BindExecutor_();
      return ReturnCodes::OK;
    }
    catch (std::bad_cast &e) {
      ;
    }

    return ReturnCodes::NO_SUITABLE_BIND_ARGS;

  }

  // Private methods //

  void MXNetServable::BindExecutor_() {

    executor_ = servable_.SimpleBind(ctx_, args_map_,
                                     std::map<std::string, mx::NDArray>(), std::map<std::string, mx::OpReqType>(),
                                     aux_map_);

    bind_called_ = true;
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

    mx::Operator("concat")(current_batch_)
            .SetParam("dim", 0)
            .SetParam("num_args", input_shape_[0])
            .Invoke(args_map_["data"]);

    executor_->Forward(false); // why does this segfault?

    mx::NDArray &result = executor_->outputs[0];
    mx::NDArray::WaitAll();

    for (auto &client_idx: idx_by_client_) {
      int client_batch_size = client_idx.second.second - client_idx.second.first;
      result_by_client_[client_idx.first] = mx::NDArray(mx::Shape(client_batch_size, output_shape_[1]), ctx_);
      result_by_client_[client_idx.first] = 0.f;
      result_by_client_[client_idx.first] += result.Slice(client_idx.second.first, client_idx.second.second);
      done_processing_by_client_.emplace(client_idx.first);
    }

    idx_by_client_.clear();

    // Reset everyone
    current_batch_.clear();
    result_cv_.notify_all();
    current_n_ = 0;

  }

} // Serving


