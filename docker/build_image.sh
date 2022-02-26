#!/bin/bash
docker build -t rangenet1.1 --network=host -f Dockerfile-tensorrt8.2.2 .

