//
// Created by Aman LaChapelle on 11/4/17.
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


#ifndef BATCHING_RPC_SERVER_TENSORBATCHINGSERVER_HPP
#define BATCHING_RPC_SERVER_TENSORBATCHINGSERVER_HPP

// STL
#include <thread>

// UUID
#include <uuid/uuid.h>

// gRPC
#include <grpc++/grpc++.h>
#include <grpc/support/log.h>

// Project
#include "Servable.hpp"

// Generated
#include "BatchingRPC.pb.h"
#include "BatchingRPC.grpc.pb.h"

namespace Serving {
  class TBServer final : public BatchingServable::Service {
  public:
    explicit TBServer(Servable *servable);
    ~TBServer() override ;
    
    grpc::Status SetBatchSize(grpc::ServerContext *ctx, const AdminRequest *req, AdminReply *rep) override ;
    grpc::Status Connect(grpc::ServerContext *ctx, const ConnectionRequest *req, ConnectionReply *rep) override ;
    grpc::Status Process(grpc::ServerContext *ctx, const TensorMessage *req, TensorMessage *rep) override ;

    void StartInsecure(const std::string &server_address);
//    void StartSSL(const std::string &server_address);
    void Stop();

  private:
    std::set<std::string> users_;
    std::thread serve_thread_;
    std::unique_ptr<grpc::Server> server_;

    Servable *servable_;
  };
}


#endif //BATCHING_RPC_SERVER_TENSORBATCHINGSERVER_HPP
