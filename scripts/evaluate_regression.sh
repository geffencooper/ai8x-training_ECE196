#!/bin/sh
./train.py --model geffennet_p2 --dataset geffen_points --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/regression_q.pth.tar -8 --device MAX78000 "$@"