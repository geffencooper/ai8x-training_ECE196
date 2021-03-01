#!/bin/sh
./train_bb_and_bc.py --lr 0.05 --epochs 50 --batch-size 128 --deterministic --compress geffen_schedule.yaml --model mini_vgg_net_bb_and_bc --dataset bb --param-hist --device MAX78000 "$@"
#./train.py --lr 0.01 --epochs 50 --deterministic --compress geffen_schedule.yaml --model mini_vgg_net_bb --dataset bb --param-hist --device MAX78000 "$@"