#!/bin/bash

VIDEO_IDX=2293
FEATURE_SET=test # train(0~10023) or test(0~4925)
EPOCH=8 # only 1~10(8 with lowest loss)

python test.py --video_idx $VIDEO_IDX --feature_set $FEATURE_SET --epoch $EPOCH