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
#include <fstream>
#include <thread>

// UUID
#include <uuid/uuid.h>

// gRPC
#include <grpc++/grpc++.h>
#include <grpc/support/log.h>

// Project
#include "Servable.hpp"

// Generated
#include <BatchingRPC.grpc.pb.h>
#include <BatchingRPC.pb.h>

/**
 * @namespace Serving
 * @brief Top level namespace for this project
 *
 * Defines the namespace within which all the code in this project resides.
 * Protobuf generates code under this namespace so for uniformity everything
 * lives under this namespace. No subdivisions exist as of 26/12/2017.
 */
namespace Serving {
/**
 * @class TBServer
 * @brief Implements the BatchingServer::Service, providing the transport/RPC
 * layer
 *
 * This class provides the transport/RPC layer to the project. This means that
 * if we implement a Servable object, we can use this class to serve requests
 * to it. The API for the Servable object can be found in Servable.hpp and the
 * API for this object can be found in BatchingRPC.proto. Requests to the
 * system as a whole must follow the API in BatchingRPC.proto and internal
 * requests from TBServer to an implementation of a Servable must follow the
 * API in Servable.hpp.
 */
class TBServer final : public BatchingServer::Service {
public:
  /**
   * @brief Constructs a new TBServer object around an already-created
   * Servable.
   *
   * @param servable A pointer to an initialized Servable object. Takes
   * ownership of the pointer upon construction.
   */
  explicit TBServer(Servable *servable);

  /**
   * @brief Destroys a TBServer object and cleans up all resources.
   */
  ~TBServer() override;

  /**
   * @brief Defines the gRPC backend for setting the batch size of the
   * Servable object. The client API for this function can be found in
   * BatchingRPC.proto
   *
   * This function calls Serving::Servable::SetBatchSize, which can return two
   * states. If an attempt is made to set the batch size to be smaller than
   * the current size of the batch, then Serving::ReturnCodes::NEXT_BATCH is
   * returned, indicating that the request should be retried. If the request
   * is successful, Serving::ReturnCodes::OK will be returned and the call can
   * proceed as normal. This call blocks on input aggregation and processing
   * of the current batch.
   *
   * @param ctx
   * @param req
   * @param rep
   * @return gRPC status to the client.
   */
  grpc::Status SetBatchSize(grpc::ServerContext *ctx, const AdminRequest *req,
                            AdminReply *rep) override;

  /**
   * @brief Defines the gRPC backend for creating a new connection to the
   * Servable. The client API for this function can be found in
   * BatchingRPC.proto
   *
   * This function creates a uuid for each client to avoid client id
   * collisions in the servable's processing space. The client should call
   * this function once, receive their unique ID, and tag all future requests
   * to Process with this unique ID. This function can be called as often as
   * desired.
   *
   * @param ctx
   * @param req
   * @param rep
   * @return gRPC status to the client.
   */
  grpc::Status Connect(grpc::ServerContext *ctx, const ConnectionRequest *req,
                       ConnectionReply *rep) override;

  /**
   * @brief Defines the gRPC backend for requesting the Servable process some
   * piece of data. The client API for this function can be found in
   * BatchingRPC.proto
   *
   * This function enqueues a request to the Servable for both data processing
   * and retreival of results. This function will block until the current
   * batch is finished processing. The client is free to call Process in
   * another thread and simply wait for the result, or use an std::async call
   * to wait for the result as the client wishes. An example of calling this
   * function in a somewhat asynchronous manner can be found in
   * TestIntegration.cpp
   *
   * @param ctx
   * @param req
   * @param rep
   * @return gRPC status to the client.
   */
  grpc::Status Process(grpc::ServerContext *ctx, const TensorMessage *req,
                       TensorMessage *rep) override;

  /**
   * @brief Starts the server at the specified address.
   *
   * Hides the gRPC server starting code behind a convenient function that can
   * be called to start the server with the credentials specified in the
   * function name.
   *
   * @param server_address Specifies the server's address - for example: @code
   * "127.0.0.1:8080" @endcode
   */
  void StartInsecure(const std::string &server_address);

  /**
   * @brief Starts the server at the specified address, with the specified
   * credentials. Note that if
   * the client does not have the correct certificate they will be unable to
   * connect.
   *
   * @param server_address Specifies the server's address - for example: @code
   * "127.0.0.1:8080" @endcode
   * @param key Either a filename or the actual key in a string. The function
   * checks for the first five dashes
   *            in the key to determine if it's a filename or not.
   * @param cert Either a filename or the actual certificate in a string. The
   * function checks for the first five dashes
   *             in the key to determine if it's a filename or not.
   */
  void StartSSL(const std::string &server_address, const std::string &key,
                const std::string &cert);

  /**
   * @brief Shuts down the server and cleans up used resources.
   */
  void Stop();

private:
  std::set<std::string> users_;
  std::thread serve_thread_;
  std::unique_ptr<grpc::Server> server_;

  std::unique_ptr<Servable> servable_;
};
} // namespace Serving

#endif // BATCHING_RPC_SERVER_TENSORBATCHINGSERVER_HPP
