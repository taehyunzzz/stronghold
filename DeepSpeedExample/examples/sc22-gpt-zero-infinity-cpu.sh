#! /bin/bash

rm -rf ./checkpoints/* 

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

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
config_json="$script_dir/ds_zero_stage_infinity-cpu.json"

# Megatron Model Parallelism
mp_size=1

NLAYERS=1
NHIDDEN=256
HEADS=16
SEQ=512
BATCHSIZE=3
LOGDIR="tensorboard_data/${NLAYERS}l_${NHIDDEN}h_${NNODES}n_${GPUS_PER_NODE}g_${mp_size}mp_${BATCHSIZE}b_ds4"

#ZeRO Configs
stage=3
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000

#Actication Checkpointing and Contigious Memory
chkp_layers=1
PA=false
PA_CPU=true
CC=false
SYNCHRONIZE=false
PROFILE=false

gpt_options=" \
        --model-parallel-size ${mp_size} \
        --num-layers $NLAYERS \
        --hidden-size $NHIDDEN \
        --num-attention-heads $HEADS \
        --seq-length $SEQ \
        --max-position-embeddings $SEQ \
        --batch-size $BATCHSIZE \
        --train-iters 50 \
        --log-interval 10 \
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
        --checkpoint-activations \
        --save-interval 10000 \
        --eval-interval 1000 \
        --eval-iters 1000
"
        #--fp16 \
        #--tensorboard-dir ${LOGDIR}
  
 deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${stage} \
                --cpu-optimizer \
                --zero-reduce-bucket-size ${rbs} \
                --zero-allgather-bucket-size ${agbs} \
                --split-transformers
            "

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-reduce-scatter"
fi

chkp_opt=" \
--checkpoint-activations \
--deepspeed-activation-checkpointing \
--checkpoint-num-layers ${chkp_layers}"

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
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --profile-backward"
fi

full_options="${gpt_options} ${deepspeed_options} ${chkp_opt}"

export PYTHONGIL=1
run_cmd="deepspeed --master_port 8895 --include localhost:0 pretrain_gpt2.py ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
