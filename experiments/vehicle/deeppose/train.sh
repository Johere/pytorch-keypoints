#!/usr/bin/env bash
set -x

ROOT=../../..
export PYTHONPATH=${ROOT}:${PYTHONPATH}

export JOBLIB_TEMP_FOLDER=/dev/shm
export TMPDIR=/dev/shm
export TORCH_HOME=/mnt/disk1/data_for_linjiaojiao/.cache
#export CUDA_VISIBLE_DEVICES='0,1'
#GPUS=${3-'0'}
#export CUDA_VISIBLE_DEVICES=${GPUS}

EXP_NAME=${1-debug}
CONFIG_FILE=${2-${ROOT}/configs/vehicle/2d_kpt_sview_rgb_img/deeppose/res50_boxcars_192x256_rle.py}
GPUID=${3-1}


OUTPUT_DIR=output/${EXP_NAME}
# for safety
if [ -d ${OUTPUT_DIR} ]; then
    echo "job:" ${EXP_NAME} "exists before, 3 seconds for your withdrawing..."
    sleep 3
    rm -rf ${OUTPUT_DIR}
fi

mkdir -p ${OUTPUT_DIR}

echo 'start training:' ${EXP_NAME} 'config:' ${CONFIG_FILE}
python ${ROOT}/tools/train.py \
    ${CONFIG_FILE} \
    --gpu-id ${GPUID} \
    --work-dir=${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/train.log

echo ${EXP_NAME} 'done.'

    # --cfg-options \
    # data.workers_per_gpu=0 \
# --resume-from='/mnt/disk1/data_for_linjiaojiao/projects/pytorch-classification/experiments/mv2/output/qnet_exp14/epoch_110.pth' \