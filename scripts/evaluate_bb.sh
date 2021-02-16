#!/bin/sh
./train.py --model geffennet_p3 --dataset geffen_bb --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/bb_q.pth.tar -8 --device MAX78000 "$@"