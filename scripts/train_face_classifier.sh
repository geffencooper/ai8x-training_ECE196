#!/bin/sh
./train.py --epochs 20 --deterministic --compress geffen_schedule.yaml --model mini_vgg_net --dataset faces_and_non_faces_80 --confusion --param-hist --pr-curves --embedding --device MAX78000 "$@"