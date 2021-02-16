#!/bin/sh
./train.py --model geffennet --dataset GEFFEN_FACES --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/classifier_q.pth.tar -8 --shap 5 --device MAX78000 "$@"