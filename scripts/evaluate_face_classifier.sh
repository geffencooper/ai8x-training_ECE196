#!/bin/sh
./train.py --model mini_vgg_net --dataset faces_and_non_faces_80 --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/test_q.pth.tar -8 --shap 5 --device MAX78000 "$@"