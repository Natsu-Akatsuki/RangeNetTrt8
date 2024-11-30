#!/bin/bash
docker build -t rangenet:ubuntu22.04_cuda12.4.1_tensorrt10.6_humble --network=host -f Dockerfile .
