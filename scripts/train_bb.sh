#!/bin/sh
./train_bb.py --lr 0.01 --epochs 50 --deterministic --compress geffen_schedule.yaml --model mini_vgg_net_bb --dataset bb --param-hist --device MAX78000 "$@"