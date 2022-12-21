#!/usr/bin/bash

cd ..
python eval/fastInferenceStage1.py $1 True $2 embed_labels amzn_labels_out.h5 Amzn13K
python eval/fastInferenceStage1.py $1 True $2 embed_instances amzn_instances_out.h5 Amzn13K
python eval/amznEvalLastStage.py amzn_labels_out.h5 amzn_instances_out.h5