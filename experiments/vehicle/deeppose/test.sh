#!/usr/bin/env bash
set -x

ROOT=../../..
DATASET_ROOT=/mnt/disk1/data_for_linjiaojiao/datasets
export PYTHONPATH=${ROOT}:${PYTHONPATH}

export JOBLIB_TEMP_FOLDER=/dev/shm
export TMPDIR=/dev/shm

EXP_NAME=${1-pose_exp0}  # pose_exp0

cfg=$(ls output/${EXP_NAME}/*.py)
CONFIG_FILE=${cfg}
TEST_SET=${2-boxcars-val}       # test, val

GPUS=${3-3}

ckpt=$(ls output/${EXP_NAME}/latest.pth)
CHECKPOINT=${ckpt}

JOB_NAME=test_${EXP_NAME}_${TEST_SET}
OUTPUT_DIR=results/${JOB_NAME}/results.json

mkdir -p ./logs

if [ ${TEST_SET} == "boxcars-val" ]; then
    DATA_TYPE=TopDownVehicleDataset
    ANN_FILE=${DATASET_ROOT}/BoxCars21k_crop/annotations/vehicle_keypoints_18points_val.json
    IMAGE_PREFIX=${DATASET_ROOT}/BoxCars21k_crop/images/
elif [ ${TEST_SET} == "boxcars-full-val" ]; then
    DATA_TYPE=TopDownVehicleDataset
    ANN_FILE=${DATASET_ROOT}/BoxCars21k/annotations/vehicle_keypoints_18points_val.json
    IMAGE_PREFIX=${DATASET_ROOT}/BoxCars21k/images/
elif [ ${TEST_SET} == "boxcars-test" ]; then
    DATA_TYPE=TopDownSimpleImageDataset
    ANN_FILE=${DATASET_ROOT}/BoxCars21k_crop/annotations/images_with_shape.list
    IMAGE_PREFIX=${DATASET_ROOT}/BoxCars21k_crop/images/
elif [ ${TEST_SET} == "uadetrac-test" ]; then
    DATA_TYPE=TopDownSimpleImageDataset
    ANN_FILE=${DATASET_ROOT}/UA_DETRAC_crop_fps5/annotations/images_with_shape.list
    IMAGE_PREFIX=${DATASET_ROOT}/UA_DETRAC_crop_fps5/images/
elif [ ${TEST_SET} == "bit-test" ]; then
    DATA_TYPE=TopDownSimpleImageDataset
    ANN_FILE=${DATASET_ROOT}/BITVehicle/annotations/cropped_images_with_shape.list
    IMAGE_PREFIX=${DATASET_ROOT}/BITVehicle/cropped_images/
fi

echo 'start testing:' ${EXP_NAME} 'config:' ${CONFIG_FILE} 'checkpoint: ' ${CHECKPOINT}
python -u ${ROOT}/tools/test.py ${CONFIG_FILE} ${CHECKPOINT} \
    --eval mAP \
    --out ${OUTPUT_DIR} \
    --gpu-id ${GPUS} \
    --cfg-options \
    data.test.type=${DATA_TYPE} \
    data.test.ann_file=${ANN_FILE} \
    data.test.img_prefix=${IMAGE_PREFIX} \
    2>&1 | tee logs/${JOB_NAME}.log

echo ${JOB_NAME} 'done.'

#    --eval mAP \
# --format-only \
