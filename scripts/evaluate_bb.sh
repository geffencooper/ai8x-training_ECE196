#!/bin/sh
./train_bb.py --model mini_vgg_net_bb --dataset bb --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/mini_vgg_net_bb_q.pth.tar -8 --device MAX78000 "$@"