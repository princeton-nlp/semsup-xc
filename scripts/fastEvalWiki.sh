#!/usr/bin/bash

cd ..
python eval/fastInferenceStage1.py $1 True $2 embed_labels wiki_labels_out.h5 Wiki1M
python eval/wikiEvalLastStage.py $1 True $2 embed_instances wiki_instances_out.h5