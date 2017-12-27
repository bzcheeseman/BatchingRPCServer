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


#ifndef BATCHING_RPC_SERVER_SERVABLE_HPP
#define BATCHING_RPC_SERVER_SERVABLE_HPP

// Generated
#include "BatchingRPC.pb.h"

/**
 * @namespace Serving
 * @brief Top level namespace for this project
 *
 * Defines the namespace within which all the code in this project resides. Protobuf generates code
 * under this namespace so for uniformity everything lives under this namespace. No subdivisions
 * exist as of 26/12/2017.
 */
namespace Serving {

  /**
   * @brief Possible return codes for a Servable object.
   *
   * These are the return codes that each Servable object may return. Some codes will only be returned by
   * certain functions.
   */
  enum ReturnCodes {
    //! Function exited normally.
    OK = 1,
    //! Bind was not called on the Servable, typically done before construction of TBServer. See Servable::Bind(BindArgs&).
    NEED_BIND_CALL = 2,
    //! The shape of the input was incorrect, check the shape and retry.
    SHAPE_INCORRECT = 3,
    //! The current batch is too full to process the request, retry.
    NEXT_BATCH = 4,
    //! The request's size is too large for the Servable's set batch size, caller should subdivide and try again
    //! or increase the batch size.
    BATCH_TOO_LARGE = 5,
    //! Bind has failed because the Servable was unable to cast the BindArgs instance to a usable type.
    NO_SUITABLE_BIND_ARGS = 6,
  };

  /**
   * @brief A virtual wrapper for arguments used to bind a Servable.
   *
   * All servables will require different arguments to bind, but they must create a subclass
   * of this object and then use a dynamic cast to cast it to the appropriate object. See an
   * example in MXNetServable.hpp - Serving::RawBindArgs and Serving::FileBindArgs.
   */
  struct BindArgs {
    virtual ~BindArgs() = default;
  };

  /**
   * @class Servable
   * @brief Delimits the public API for a Servable object.
   *
   * In order to be used with a TBServer gRPC wrapper, a Servable must implement this API.
   */
  class Servable {
  public:
    virtual ~Servable() = default;

    /**
     * @brief Sets the batch size of the Servable.
     *
     * The Servable may have a need to have the size of its batches changed during runtime - this function allows
     * this.
     *
     * @param new_size The new batch size of the servable.
     * @return Returns ReturnCodes::OK if successful or ReturnCodes::NEXT_BATCH if the batch is already
     *         larger than new_size.
     */
    virtual ReturnCodes SetBatchSize(const int &new_size) = 0;

    /**
     * @brief Adds the TensorMessage to the batch
     *
     * Deserializes the TensorMessage and adds it to the batch. Internally tracks the indices to return the
     * correct result to each client. It is important to note that the serialization of the TensorMessage
     * on the client side and on the Servable must match for results to make any sense - a client should
     * serialize in MXNet format for a MXNetServable, for example.
     *
     * @param message The TensorMessage we are requesting to process.
     * @return Returns any of [ReturnCodes::OK, ReturnCodes::NEED_BIND_CALL,
     *         ReturnCodes::SHAPE_INCORRECT, ReturnCodes::NEXT_BATCH, ReturnCodes::BATCH_TOO_LARGE].
     */
    virtual ReturnCodes AddToBatch(const TensorMessage &message) = 0;

    /**
     * @brief Gets the client's result. Blocks until the result is available.
     *
     * Finds client_id's result and stores it in message with implementation-specific serialization. It is
     * important to note that the serialization of the TensorMessage on the client side and on the
     * Servable must match for results to make any sense - a client should  serialize in MXNet format for
     * a MXNetServable, for example.
     *
     * @param client_id A string containing a unique client identifier whose result we want to fetch.
     * @param message An initialized TensorMessage that we can store the data inside of. The caller assumes
     *                responsibility for memory management.
     * @return
     */
    virtual ReturnCodes GetResult(const std::string &client_id, TensorMessage *message) = 0;

    /**
     * @brief Bind the algorithm to the Servable.
     *
     * Binds the algorithm to be served to the Servable. This will have implementation specific details that means
     * the BindArgs implementations will vary wildly. MXNetServable::Bind(BindArgs&) has an example.
     *
     * @param args The algorithm, whether it's in a variable or stored in a file.
     * @return Returns either ReturnCodes::OK if successful or ReturnCodes::NO_SUITABLE_BIND_ARGS if the cast
     *         is unsuccessful.
     */
    virtual ReturnCodes Bind(BindArgs &args) = 0;
  };
} // Serving

#endif //BATCHING_RPC_SERVER_SERVABLE_HPP
