#!/bin/bash

cd "$(dirname $0)"

docker-compose run -p 8888:8888 --rm -u root sd-infinity bash
