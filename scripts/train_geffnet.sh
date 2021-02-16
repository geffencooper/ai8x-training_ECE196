#!/bin/sh
./train.py --epochs 20 --compress geffen_schedule.yaml --model geffnet --dataset geffnet_faces --confusion --param-hist --pr-curves --embedding --save-sample 0 --device MAX78000 "$@"