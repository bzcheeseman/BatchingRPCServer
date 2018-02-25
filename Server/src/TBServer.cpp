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
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::ServerContext;
using grpc::Status;

namespace {
std::string ReadFile_(const std::string &filename) {
  std::ifstream in_file;

  std::string tmp, out;
  in_file.open(filename);
  while (in_file.good()) {
    std::getline(in_file, tmp);
    out += tmp + "\n";
  }
  in_file.close();

  return out;
}
}

namespace Serving {

TBServer::TBServer(Servable *servable) : servable_(servable) { ; }

TBServer::~TBServer() { ; }

grpc::Status TBServer::SetBatchSize(grpc::ServerContext *ctx,
                                    const AdminRequest *req, AdminReply *rep) {
  ReturnCodes code = servable_->SetBatchSize(req->new_batch_size());

  switch (code) {
  case OK:
    break;
  case NEXT_BATCH: {
    grpc::Status early_exit_status(
        grpc::UNAVAILABLE,
        "Batch is already larger than requested size, retry");
    return early_exit_status;
  }
  default: {
    grpc::Status early_exit_status(grpc::CANCELLED,
                                   "An error ocurred, try again later");
    return early_exit_status;
  }
  }

  return grpc::Status::OK;
}

Status TBServer::Connect(ServerContext *ctx, const ConnectionRequest *req,
                         ConnectionReply *rep) {

  uuid_t uuid;
  uuid_generate(uuid);
  char uuid_str[37];
  uuid_unparse_lower(uuid, uuid_str);
  users_.emplace(uuid_str);

  rep->set_client_id(uuid_str);

  return Status::OK;
}

grpc::Status TBServer::Process(ServerContext *ctx, const TensorMessage *req,
                               TensorMessage *rep) {

  auto user = users_.find(req->client_id());
  if (user == users_.end()) {
    grpc::Status early_exit_status(grpc::FAILED_PRECONDITION,
                                   "Connect not called, client id unknown");
    return early_exit_status;
  }

  // TODO: make sure that this is all going to the same instance
  ReturnCodes code = servable_->AddToBatch(*req); // Add to batch and move on

  switch (code) {
  case OK:
    break;
  case NEED_BIND_CALL: {
    grpc::Status early_exit_status(grpc::FAILED_PRECONDITION,
                                   "Bind not called on servable");
    return early_exit_status;
  }
  case SHAPE_INCORRECT: {
    grpc::Status early_exit_status(grpc::INVALID_ARGUMENT,
                                   "Input tensor shape incorrect");
    return early_exit_status;
  }
  case NEXT_BATCH: {
    grpc::Status early_exit_status(grpc::UNAVAILABLE,
                                   "Attempted to add to already full batch");
    return early_exit_status;
  }
  case BATCH_TOO_LARGE: {
    grpc::Status early_exit_status(
        grpc::INVALID_ARGUMENT,
        "Batch request was too large, split into smaller pieces and retry");
    return early_exit_status;
  }
  case NO_SUITABLE_BIND_ARGS:
    break; // this one won't be thrown by the function
  }

  code = servable_->GetResult(req->client_id(), rep);

  switch (code) {
  case OK:
    break;
  case NEXT_BATCH: {
    grpc::Status early_exit_status(
        grpc::UNAVAILABLE, "Try again later, processing hasn't yet started!");
    return early_exit_status;
  }
  default: {
    grpc::Status early_exit_status(grpc::CANCELLED,
                                   "An error ocurred, try again later");
    return early_exit_status;
  }
  }

  return grpc::Status::OK;
}

void TBServer::StartInsecure(const std::string &server_address) {
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(this);
  server_ = builder.BuildAndStart();

  serve_thread_ = std::thread([&]() { server_->Wait(); });
}

void TBServer::Stop() {
  server_->Shutdown();
  serve_thread_.join();
}

void TBServer::StartSSL(const std::string &server_address,
                        const std::string &key, const std::string &cert) {
  ServerBuilder builder;

  bool key_dashes = true;
  bool cert_dashes = true;
  for (uint16_t i = 0; i < 5; i++) {
    key_dashes &= key[i] == '-';
    cert_dashes &= cert[i] == '-';
  }

  grpc::SslServerCredentialsOptions::PemKeyCertPair pkcp;
  if (!key_dashes && !cert_dashes)
    pkcp = {ReadFile_(key), ReadFile_(cert)};
  if (!key_dashes && cert_dashes)
    pkcp = {ReadFile_(key), cert};
  if (key_dashes && !cert_dashes)
    pkcp = {key, ReadFile_(cert)};
  if (key_dashes && cert_dashes)
    pkcp = {key, cert};

  grpc::SslServerCredentialsOptions ssl_opts;
  ssl_opts.pem_root_certs = "";
  ssl_opts.pem_key_cert_pairs.push_back(pkcp);

  std::shared_ptr<grpc::ServerCredentials> channel_creds =
      grpc::SslServerCredentials(ssl_opts);
  builder.AddListeningPort(server_address, channel_creds);
  builder.RegisterService(this);
  server_ = builder.BuildAndStart();

  serve_thread_ = std::thread([&]() { server_->Wait(); });
}

} // namespace Serving
