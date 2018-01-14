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

#ifndef BATCHINGRPCSERVER_DLIBSERVABLE_HPP
#define BATCHINGRPCSERVER_DLIBSERVABLE_HPP

// STL
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

// Dlib
#include "dlib/dnn.h"

// Project
#include "Servable.hpp"

// Generated
#include "BatchingRPC.pb.h"

namespace Serving {

  struct DlibFileBindArgs : public BindArgs {
    std::string filename;
  };

  template<class NetType>
  struct DlibRawBindArgs : public BindArgs {
    NetType net;
  };

  template<
      class NetType,
      class InputType, // input type is dlib::matrix of dlib::rgb_pixel, float,
                       // unsigned char, etc.
      typename OutputType // output type is anything serializable by dlib
      >
  class DlibServable : public Servable {
  public:
    DlibServable(const int &batch_size);
    ~DlibServable() override;

    ReturnCodes SetBatchSize(const int &new_size) override;

    ReturnCodes AddToBatch(const TensorMessage &message) override;

    ReturnCodes
    GetResult(const std::string &client_id, TensorMessage *message) override;

    ReturnCodes Bind(BindArgs &args) override;

  private:
    void SetBatchSize_(const int &new_size);
    void ProcessCurrentBatch_();

  private:
    NetType servable_;

    std::mutex input_mutex_;
    std::map<std::string, std::pair<int, int>> idx_by_client_;
    std::vector<InputType> current_batch_;

    int current_n_;
    int batch_size_;

    std::atomic<bool> bind_called_;

    std::mutex result_mutex_;
    std::condition_variable result_cv_;
    std::set<std::string> done_processing_by_client_;
    std::map<std::string, std::vector<OutputType>> result_by_client_;
  };

  // Implementation

  template<class NetType, class InputType, class OutputType>
  DlibServable<NetType, InputType, OutputType>::DlibServable(
      const int &batch_size) {
    servable_ = NetType();
    current_n_ = 0;
    batch_size_ = batch_size;
    bind_called_ = false;
    current_batch_.reserve(batch_size_);
  }

  template<class NetType, class InputType, class OutputType>
  DlibServable<NetType, InputType, OutputType>::~DlibServable() {
    ;
  }

  template<class NetType, class InputType, class OutputType>
  ReturnCodes DlibServable<NetType, InputType, OutputType>::SetBatchSize(
      const int &new_size) {
    std::lock_guard<std::mutex> guard_input(input_mutex_);

    if (new_size <= current_n_) {
      return ReturnCodes::NEXT_BATCH;
    }

    this->SetBatchSize_(new_size);

    return ReturnCodes::OK;
  }

  template<class NetType, class InputType, class OutputType>
  ReturnCodes DlibServable<NetType, InputType, OutputType>::AddToBatch(
      const TensorMessage &message) {
    const std::string &client_id = message.client_id();
    std::vector<InputType> message_input(message.n());
    std::istringstream message_stream(message.serialized_buffer(), std::ios::binary);

    if (!bind_called_) {
      return ReturnCodes::NEED_BIND_CALL;
    }

    if (message.n() > batch_size_) {
      return ReturnCodes::BATCH_TOO_LARGE;
    }

    // error deserializing object of type unsigned long while deserializing object of type std::vector(?)
    dlib::deserialize(message_input, message_stream);

    if (message_input.size() != message.n()) {
      return ReturnCodes::SHAPE_INCORRECT;
    }

    {

      std::lock_guard<std::mutex> guard_input(input_mutex_);

      if (message.n() + current_n_ > batch_size_) {
        this->SetBatchSize_(current_n_);
        ProcessCurrentBatch_();
        return ReturnCodes::NEXT_BATCH;
      }

      result_by_client_.erase(client_id); // clears room for the new result

      if (idx_by_client_.find(client_id) == idx_by_client_.end()) {
        idx_by_client_[client_id] =
            std::make_pair(current_n_, current_n_ + message.n());
      } else {
        idx_by_client_[client_id].second += message.n();
      }

      // Clients could send us multiple inputs, we need to deal with that properly.
      current_batch_.insert(current_batch_.end(), message_input.begin(), message_input.end());

      current_n_ += message.n();
      if (current_n_ <= batch_size_) {
        if (current_n_ == batch_size_) {
          ProcessCurrentBatch_();
        }
      }
    }

    return ReturnCodes::OK;
  }

  template<class NetType, class InputType, class OutputType>
  ReturnCodes DlibServable<NetType, InputType, OutputType>::GetResult(
      const std::string &client_id, TensorMessage *message) {
    std::unique_lock<std::mutex> lk(result_mutex_);
    result_cv_.wait(
        lk, [&, this]() { // This will block forever if the client only calls
                          // GetResult and never AddToBatch
          auto done = done_processing_by_client_.find(client_id);
          if (done != done_processing_by_client_.end()) {
            done = done_processing_by_client_.erase(done);
            return true;
          } else {
            return false;
          }
        });

    auto result = result_by_client_.find(client_id);
    std::vector<OutputType> &result_array = result->second;

    std::stringstream out_stream;
    dlib::serialize(result_array, out_stream);
    message->set_serialized_buffer(out_stream.str());
    message->set_client_id(client_id);

    result = result_by_client_.erase(result);

    return ReturnCodes::OK;
  }

  template<class NetType, class InputType, class OutputType>
  ReturnCodes
  DlibServable<NetType, InputType, OutputType>::Bind(BindArgs &args) {
    try {
      DlibFileBindArgs &file_args = dynamic_cast<DlibFileBindArgs &>(args);
      dlib::deserialize(file_args.filename) >> servable_;
      bind_called_ = true;
      return ReturnCodes::OK;
    } catch (std::bad_cast &e) {
      ;
    }
    try {
      DlibRawBindArgs<NetType> &raw_args =
          dynamic_cast<DlibRawBindArgs<NetType> &>(args);
      servable_ = std::move(raw_args.net);
      bind_called_ = true;
      return ReturnCodes::OK;
    } catch (std::bad_cast &e) {
      ;
    }

    return ReturnCodes::NO_SUITABLE_BIND_ARGS;
  }

  template<class NetType, class InputType, class OutputType>
  void DlibServable<NetType, InputType, OutputType>::SetBatchSize_(
      const int &new_size) {
    batch_size_ = new_size;
    current_batch_.reserve(batch_size_);
  }

  template<class NetType, class InputType, typename OutputType>
  void DlibServable<NetType, InputType, OutputType>::ProcessCurrentBatch_() {
    std::vector<OutputType> outputs = servable_(current_batch_);

    for (auto &client_idx : idx_by_client_) {
      result_by_client_[client_idx.first] = std::vector<OutputType>(
          outputs.begin() + client_idx.second.first,
          outputs.begin() + client_idx.second.second);
      done_processing_by_client_.emplace(client_idx.first);
    }

    idx_by_client_.clear();

    // Reset everyone
    current_batch_.clear();
    result_cv_.notify_all();
    current_n_ = 0;
  }

} // namespace Serving

#endif // BATCHINGRPCSERVER_DLIBSERVABLE_HPP
