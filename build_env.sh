#!/usr/bin/env bash

cd buildenv && docker build -t batchingrpcserver-env .
docker tag batchingrpcserver-env bzcheeseman/batchingrpcserver-env
docker push