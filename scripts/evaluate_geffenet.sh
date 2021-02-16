#!/bin/sh
./train.py --model geffnet --dataset geffnet_faces --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/geffnet_q.pth.tar -8 --device MAX78000 "$@"