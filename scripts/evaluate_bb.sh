#!/bin/sh
./train.py --model mini_vgg_net_bb --dataset bb --evaluate --batch-size 128 --exp-load-weights-from ../ai8x-synthesis/trained/bb_q.pth.tar -8 --device MAX78000 "$@"