#!/bin/sh
./train.py --model geffennet --dataset GEFFEN_FACES --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/geffen_classifier_q.pth.tar -8 --device MAX78000 "$@"