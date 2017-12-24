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


#include "TBServer.hpp"

using grpc::Server;
using grpc::ServerContext;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;

namespace Serving {

  TBServer::TBServer(
          Servable *servable
  ): servable_(servable) {
    ;
  }

  TBServer::~TBServer() {
    ;
  }

  grpc::Status TBServer::SetBatchSize(grpc::ServerContext *ctx, const AdminRequest *req, AdminReply *rep) {
    ReturnCodes code = servable_->UpdateBatchSize(req->new_batch_size());

    switch (code) {
      case OK: break;
      case NEXT_BATCH: {
        grpc::Status early_exit_status (grpc::UNAVAILABLE, "Batch is already larger than requested size, retry");
        return early_exit_status;
      }
    }

    return grpc::Status::OK;
  }

  Status TBServer::Connect(ServerContext *ctx, const ConnectionRequest *req, ConnectionReply *rep) {

    uuid_t uuid;
    uuid_generate(uuid);
    char uuid_str[37];
    uuid_unparse_lower(uuid, uuid_str);
    users_.emplace(uuid_str);

    rep->set_client_id(uuid_str);

    return Status::OK;
  }

  grpc::Status TBServer::Process(ServerContext *ctx, const TensorMessage *req, TensorMessage *rep) {

    auto user = users_.find(req->client_id());
    if (user == users_.end()) {
      grpc::Status early_exit_status (grpc::FAILED_PRECONDITION, "Connect not called, client id unknown");
      return early_exit_status;
    }

    // TODO: make sure that this is all going to the same servable - only one instance!
    ReturnCodes code = servable_->AddToBatch(*req); // Add to batch and move to the next stage

    switch (code) {
      case OK: break;
      case NEED_BIND_CALL: {
        grpc::Status early_exit_status (grpc::FAILED_PRECONDITION, "Bind not called on servable");
        return early_exit_status;
      }
      case SHAPE_INCORRECT: {
        grpc::Status early_exit_status (grpc::INVALID_ARGUMENT, "Input tensor shape incorrect");
        return early_exit_status;
      }
      case NEXT_BATCH: {
        grpc::Status early_exit_status (grpc::UNAVAILABLE, "Attempted to add to already full batch");
        return early_exit_status;
      }
      case BATCH_TOO_LARGE: {
        grpc::Status early_exit_status (grpc::INVALID_ARGUMENT, "Batch request was too large, split into smaller pieces and retry");
        return early_exit_status;
      }
    }

    *rep = servable_->GetResult(req->client_id());
    return grpc::Status::OK;
  }

  void TBServer::StartInsecure(const std::string &server_address) {
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(this);
    server_ = builder.BuildAndStart();

    serve_thread_ = std::thread([&](){server_->Wait();});
  }

  void TBServer::Stop() {
    server_->Shutdown();
    serve_thread_.join();
  }

//  void TBServer::StartSSL(const std::string &server_address) {
//    ServerBuilder builder;
//    builder.AddListeningPort(server_address, grpc::SslServerCredentials());
//    builder.RegisterService(this);
//    server_ = builder.BuildAndStart();
//
//    serve_thread_ = std::thread([&](){server_->Wait();});
//  }

}
