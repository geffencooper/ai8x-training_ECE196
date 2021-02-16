#!/bin/sh
./train.py --epochs 20 --deterministic --compress geffen_schedule.yaml --model geffennet_p2 --dataset geffen_points --param-hist --device MAX78000 "$@"