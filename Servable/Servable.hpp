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

#include "BatchingRPC.pb.h"

namespace Serving {
  enum ReturnCodes {
    OK = 1,
    NEED_BIND_CALL = 2,
    SHAPE_INCORRECT = 3,
    NEXT_BATCH = 4,
    BATCH_TOO_LARGE = 5,
  };

  class Servable {
  public:
    virtual ~Servable() = 0;
    virtual ReturnCodes AddToBatch(const TensorMessage &message, std::string client_id) = 0;
    virtual TensorMessage GetResult(std::string client_id) = 0;
  };

  Servable::~Servable() {
    ;
  }
} // Serving

#endif //BATCHING_RPC_SERVER_SERVABLE_HPP
