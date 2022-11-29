#!/usr/bin/env bash
set -x

ROOT=../..
export PYTHONPATH=${ROOT}:${PYTHONPATH}

export JOBLIB_TEMP_FOLDER=/dev/shm
export TMPDIR=/dev/shm
export TORCH_HOME=/mnt/disk1/data_for_linjiaojiao/.cache

# # to debug NCCL error
# export NCCL_DEBUG=WARN


EXP_NAME=${1-debug}
GPUS=${2-'0,1,2,3'}
# get gpu-nums
gpu_ids=(${GPUS//,/ })
ngpus=${#gpu_ids[@]}
export CUDA_VISIBLE_DEVICES=${GPUS}

CONFIG_FILE=${3-${ROOT}/configs/mobilenet_v2/0.5mobilenet-v2_regression.py}

EXP_NAME=${EXP_NAME}_${ngpus}gpus
OUTPUT_DIR=output/${EXP_NAME}
# for safety
if [ -d ${OUTPUT_DIR} ]; then
    echo "job:" ${EXP_NAME} "exists before, 3 seconds for your withdrawing..."
    sleep 3
    rm -rf ${OUTPUT_DIR}
fi

mkdir -p ${OUTPUT_DIR}

echo 'start training:' ${EXP_NAME} 'config:' ${CONFIG_FILE}

python -m torch.distributed.launch --nproc_per_node=${ngpus} --master_port=29511 \
${ROOT}/tools/train.py \
    ${CONFIG_FILE} \
    --launcher pytorch \
    --work-dir=${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/train.log

echo ${EXP_NAME} 'done.'
