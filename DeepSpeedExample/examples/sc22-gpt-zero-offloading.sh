#! /bin/bash

rm -rf ./checkpoints/*

GPUS_PER_NODE=1
# Change for multinode config
export MASTER_ADDR=localhost
export MASTER_PORT=8891
export RANK=0
export LOCAL_RANK=0
export NNODES=1
export NODE_RANK=0
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export DLWS_NUM_WORKER=${NNODES}
export DLWS_NUM_GPU_PER_WORKER=${GPUS_PER_NODE}

_BASE=/home/kimth/workspace/stronghold
# DATA_PATH=${_BASE}/data/openwebtext/openwebtext.txt
DATA_PATH=${_BASE}/data/wikitext-2-v1/preprocessed_text_document
VOCAB_PATH=${_BASE}/gpt2config/gpt2-vocab.json
MERGE_PATH=${_BASE}/gpt2config/gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_ds

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
config_json="$script_dir/ds_zero_stage_2_config.json"

# Megatron Model Parallelism
mp_size=1

# NLAYERS=${1-24}
# NHIDDEN=${2-2560}
# HEADS=${3-16}
# SEQ=${4-1024}
# BATCHSIZE=${5-4}
NLAYERS=50
NHIDDEN=1024
HEADS=16
SEQ=1024
# BATCHSIZE=4
LOGDIR="tensorboard_data/${NLAYERS}l_${NHIDDEN}h_${NNODES}n_${GPUS_PER_NODE}g_${mp_size}mp_${BATCHSIZE}b_ds4"

#ZeRO Configs
stage=2
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000

#Actication Checkpointing and Contigious Memory
# chkp_layers=2
# PA=true
PA=false
# PA_CPU=true
# PA_CPU=false
# CC=true
CC=false
SYNCHRONIZE=false
PROFILE=false
ITERATIONS=3

for BATCHSIZE in 4 16 32
do
for chkp_layers in 0 1 20
do
for PA_CPU in false true
do

if [ "${chkp_layers}" = "0" ]; then
        # PA_CPU=false
        echo "CPU activation checkpoint disabled"
else 
        # PA_CPU=true
        echo "CPU activation checkpoint enabled"

        chkp_opt=" \
        --checkpoint-activations \
        --deepspeed-activation-checkpointing \
        --checkpoint-num-layers ${chkp_layers} \
        --split-transformers"

        if [ "${PA}" = "true" ]; then
        chkp_opt="${chkp_opt} \
                --partition-activations"
        fi

        if [ "${PA_CPU}" = "true" ]; then
        chkp_opt="${chkp_opt} \
                --checkpoint-in-cpu"
        fi

        if [ "${SYNCHRONIZE}" = "true" ]; then
        chkp_opt="${chkp_opt} \
                --synchronize-each-layer"
        fi

        if [ "${CC}" = "true" ]; then
        chkp_opt="${chkp_opt} \
                --contiguous-checkpointing"
        fi

        if [ "${PROFILE}" = "true" ]; then
        chkp_opt="${chkp_opt} \
                --profile-backward"
        fi
fi

gpt_options=" \
        --model-parallel-size ${mp_size} \
        --num-layers $NLAYERS \
        --hidden-size $NHIDDEN \
        --num-attention-heads $HEADS \
        --seq-length $SEQ \
        --max-position-embeddings $SEQ \
        --batch-size $BATCHSIZE \
        --train-iters $ITERATIONS \
        --log-interval 1 \
        --exit-interval 50 \
        --lr-decay-iters 320000 \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_PATH \
        --merge-file $MERGE_PATH \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 1.5e-4 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup 0.01 \
        --save-interval 10000 \
        --eval-interval 1000 \
        --eval-iters 1000
"
  
 deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${stage} \
                --cpu-optimizer \
                --zero-reduce-bucket-size ${rbs} \
                --zero-allgather-bucket-size ${agbs} 
            "

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-reduce-scatter"
fi

full_options="${gpt_options} ${deepspeed_options} ${chkp_opt}"

export PYTHONGIL=1
# run_cmd="deepspeed --num_nodes ${DLWS_NUM_WORKER} --num_gpus ${DLWS_NUM_GPU_PER_WORKER} pretrain_gpt2.py ${full_options}"
run_cmd="deepspeed --master_port 8892 --include localhost:1 pretrain_gpt2.py ${full_options}"
echo ${run_cmd}
# eval ${run_cmd}
eval ${run_cmd} > log_ckpt/log_BATCH${BATCHSIZE}_CKPT${chkp_layers}_CPU${PA_CPU}.txt

set +x

done
done
done