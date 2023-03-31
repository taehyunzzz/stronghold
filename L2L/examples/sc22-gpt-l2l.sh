#! /bin/bash

# Runs the "345M" parameter model
rm -rf ./checkpoints/*

RANK=0
WORLD_SIZE=1

export PYTORCH_JIT=0
export MASTER_ADDR=localhost
export MASTER_PORT=6000

_BASE=/home/kimth/workspace/stronghold
# DATA_PATH=${_BASE}/data/openwebtext/openwebtext.txt
DATA_PATH=${_BASE}/data/wikitext-2-v1/preprocessed_text_document
VOCAB_PATH=${_BASE}/gpt2config/gpt2-vocab.json
MERGE_PATH=${_BASE}/gpt2config/gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_ds

# NLAYERS=${1-24}
# NHIDDEN=${2-2560}
# HEADS=${3-16}
# SEQ=${4-1024}
# BATCHSIZE=${5-4}
NLAYERS=10
NHIDDEN=1024
HEADS=16
SEQ=1024
BATCHSIZE=64
MICROBATCHSIZE=16
LOGDIR="tensorboard_data/${NLAYERS}l_${NHIDDEN}h_${NNODES}n_${GPUS_PER_NODE}g_${mp_size}mp_${BATCHSIZE}b_ds4"

for BATCHSIZE in 128 256
do
for MICROBATCHSIZE in 128
do
echo Running for MB${MICROBATCHSIZE}_B${BATCHSIZE}
PYTHONGIL=1 python pretrain_gpt.py \
       --num-layers ${NLAYERS} \
       --hidden-size ${NHIDDEN} \
       --num-attention-heads ${HEADS} \
       --micro-batch-size ${MICROBATCHSIZE} \
       --global-batch-size ${BATCHSIZE} \
       --seq-length ${SEQ} \
       --max-position-embeddings ${SEQ} \
       --train-iters 5 \
       --log-interval 1 \
       --exit-interval 50 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file ${VOCAB_PATH} \
       --merge-file ${MERGE_PATH} \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --activations-checkpoint-method uniform \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 1 \
       --enable-l2l > log/tmp_MB${MICROBATCHSIZE}_B${BATCHSIZE}.txt
       sleep 3
done
done
