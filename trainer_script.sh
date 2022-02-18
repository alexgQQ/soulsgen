#!/bin/bash

# Startup script for a trainer vm
# Delay the script to allow Nvidia drivers to initialize
# and use the GCP deeptrainer cuda python runtime

# these comments will be removed (startup scripts have a size limit)
# and the storage bucket template filled on deploy i.e. `manage create trainer`
# Access cloud logging to observe training progress

sleep 300
gsutil cp $BUCKET/models/soulsgen/latest/corpus.zip .
unzip corpus.zip
git clone https://github.com/alexgQQ/GPT2.git
cd GPT2
/opt/conda/bin/pip install -r requirements.txt
cd src
/opt/conda/bin/python -m gpt2 train --train_corpus /dataset/build/corpus.train.txt \
                       --eval_corpus /dataset/build/corpus.test.txt \
                       --vocab_path /dataset/build/vocab.txt \
                       --save_checkpoint_path ckpt.pth \
                       --save_model_path latest.pth \
                       --batch_train 128 \
                       --batch_eval 128 \
                       --seq_len 64 \
                       --total_steps 1000 \
                       --eval_steps 100 \
                       --save_steps 100 --gpus 1
gsutil cp gpt2-pretrained.pth $BUCKET/models/soulsgen/latest/model.pth
