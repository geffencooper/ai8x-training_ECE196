#!/bin/sh
./train.py --epochs 20 --deterministic --compress geffen_schedule.yaml --model geffennet --dataset GEFFEN_FACES --confusion --param-hist --pr-curves --embedding --device MAX78000 "$@"