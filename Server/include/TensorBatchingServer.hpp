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


#ifndef RPC_BATCH_SCHEDULER_TENSORBATCHINGSERVER_HPP
#define RPC_BATCH_SCHEDULER_TENSORBATCHINGSERVER_HPP

#include <uuid/uuid.h>

#include <grpc++/grpc++.h>
#include <grpc/support/log.h>
#include <thread>
#include "Servable.hpp"

#include "TensorMessage.pb.h"
#include "TensorMessage.grpc.pb.h"

namespace Serving {
  class TBServer final {
  public:
    explicit TBServer(Servable *servable);

    ~TBServer();

    void Run(const std::string &server_address);

  private:

    void Handle_();

    std::unique_ptr<grpc::ServerCompletionQueue> cq_;
    BatchingServable::AsyncService service_;
    std::unique_ptr<grpc::Server> server_;

    Servable *servable_;
  };
}


#endif //RPC_BATCH_SCHEDULER_TENSORBATCHINGSERVER_HPP
