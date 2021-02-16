#!/bin/sh
./train.py --lr 0.01 --epochs 50 --deterministic --compress geffen_schedule.yaml --model geffennet_p3 --dataset geffen_bb --param-hist --device MAX78000 "$@"