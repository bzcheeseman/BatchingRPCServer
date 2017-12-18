//
// Created by Aman LaChapelle on 11/4/17.
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


#ifndef RPC_BATCH_SCHEDULER_SERVABLE_HPP
#define RPC_BATCH_SCHEDULER_SERVABLE_HPP

#include "TensorMessage.pb.h"

namespace Serving { namespace internal {
  enum ReturnCodes {
    OK = 1,
    NEED_BIND_CALL = 2,
    SHAPE_INCORRECT = 3,
    NEXT_BATCH = 4,
    BATCH_TOO_LARGE = 5,
  };

  class Servable {
  public:
    virtual ReturnCodes AddToBatch(TensorMessage &message, std::string client_id) = 0;
    virtual TensorMessage GetResult(std::string client_id) = 0;
  };
}} // Serving::internal

#endif //RPC_BATCH_SCHEDULER_SERVABLE_HPP
