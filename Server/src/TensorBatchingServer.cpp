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


#include "TensorBatchingServer.hpp"

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;

namespace Serving {

  class CallData {
  public:
    virtual void Proceed() = 0;
  };

  class ProcessCallData : public CallData {
  public:
    ProcessCallData(
            BatchingServable::AsyncService *service,
            grpc::ServerCompletionQueue *cq,
            Servable *servable
    ) : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE), servable_(servable) {

      auto cid_iter = ctx_.client_metadata().find("client_id");
      if (cid_iter == ctx_.client_metadata().end()) {
        grpc::Status early_exit_status (grpc::FAILED_PRECONDITION, "No client metadata found!");
        responder_.FinishWithError(early_exit_status, this);
      }

      // We get the client's uuid from the map
      client_id_ = std::string(cid_iter->second.begin(), cid_iter->second.end());

      Proceed();

    }

    void Proceed() override {
      if (status_ == CREATE) {
        status_ = PROCESS;

        // Request a tag for the Process call
        service_->RequestProcess(&ctx_, &request_, &responder_, cq_, cq_, this);

      } else if (status_ == PROCESS) {
        status_ = RESULT;

        new ProcessCallData(service_, cq_, servable_);

        ReturnCodes code = servable_->AddToBatch(request_, client_id_); // Add to batch and move to the next stage

        switch (code) {
          case OK: break;
          case NEED_BIND_CALL: {
            grpc::Status early_exit_status (grpc::FAILED_PRECONDITION, "Bind not called on servable");
            status_ = FINISH;
            responder_.FinishWithError(early_exit_status, this);
            break;
          }
          case SHAPE_INCORRECT: {
            grpc::Status early_exit_status (grpc::INVALID_ARGUMENT, "Input tensor shape incorrect");
            status_ = FINISH;
            responder_.FinishWithError(early_exit_status, this);
            break;
          }
          case NEXT_BATCH: {
            grpc::Status early_exit_status (grpc::UNAVAILABLE, "Attempted to add to already full batch");
            status_ = FINISH;
            responder_.FinishWithError(early_exit_status, this);
            break;
          }
          case BATCH_TOO_LARGE: {
            grpc::Status early_exit_status (grpc::INVALID_ARGUMENT, "Batch request was too large, split into smaller pieces and retry");
            status_ = FINISH;
            responder_.FinishWithError(early_exit_status, this);
            break;
          }
        }

      } else if (status_ == RESULT) { // I want my result now plz
        status_ = FINISH;

        reply_ = servable_->GetResult(client_id_); // this is a blocking call (for now at least)

        responder_.Finish(reply_, grpc::Status::OK, this);
      } else {
        GPR_ASSERT(status_ == FINISH);
        delete this;
      }
    }

  private:
    BatchingServable::AsyncService* service_;
    grpc::ServerCompletionQueue* cq_;
    grpc::ServerContext ctx_;

    std::string client_id_;

    TensorMessage request_;
    TensorMessage reply_;

    grpc::ServerAsyncResponseWriter<TensorMessage> responder_;

    Servable *servable_;

    enum CallStatus { CREATE, PROCESS, RESULT, FINISH };
    CallStatus status_;
  };

  class ConnectCallData: public CallData { // receive request and simply return to them a uuid
  public:
    ConnectCallData(
            BatchingServable::AsyncService *service,
            grpc::ServerCompletionQueue *cq
    ) : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE) {
      Proceed();
    }

    void Proceed() override {
      if (status_ == CREATE) {
        status_ = PROCESS;

        // Request a tag for the Process call
        service_->RequestConnect(&ctx_, &request_, &responder_, cq_, cq_, this);

      } else if (status_ == PROCESS) {

        status_ = FINISH;

        new ConnectCallData(service_, cq_);

        uuid_t uuid;
        uuid_generate(uuid);
        char uuid_str[37];
        uuid_unparse_lower(uuid, uuid_str);

        reply_.set_client_id(uuid_str);

        responder_.Finish(reply_, grpc::Status::OK, this);

      } else {
        GPR_ASSERT(status_ == FINISH);
        delete this;
      }

    }

  private:
    BatchingServable::AsyncService* service_;
    grpc::ServerCompletionQueue* cq_;
    grpc::ServerContext ctx_;

    ConnectionRequest request_;
    ConnectionReply reply_;

    grpc::ServerAsyncResponseWriter<ConnectionReply> responder_;

    enum CallStatus { CREATE, PROCESS, FINISH };
    CallStatus status_;
  };

  TBServer::TBServer(
          Servable *servable
  ): servable_(servable) {
    ;
  }

  TBServer::~TBServer() {
    server_->Shutdown();
    cq_->Shutdown();
  }

  void TBServer::Run(const std::string &server_address) {
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service_);
    cq_ = builder.AddCompletionQueue();
    server_ = builder.BuildAndStart();

    Handle_();
  }

  void TBServer::Handle_() {

    new ProcessCallData(&service_, cq_.get(), servable_);
    new ConnectCallData(&service_, cq_.get());

    void *tag;
    bool ok;

    while (true) {
      GPR_ASSERT(cq_->Next(&tag, &ok));
      GPR_ASSERT(ok);

      static_cast<CallData *>(tag)->Proceed();
    }

  }

}
