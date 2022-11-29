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

JOB_NAME=test_${EXP_NAME}_${TEST_SET}_with-filter
# JOB_NAME=test_${EXP_NAME}_${TEST_SET}
OUTPUT_FILE=results/${JOB_NAME}/results.json

mkdir -p ./logs

if [ ${TEST_SET} == "boxcars-val" ]; then
    DATA_TYPE=TopDownVehicleDataset
    ANN_FILE=${DATASET_ROOT}/BoxCars21k_crop/annotations/vehicle_keypoints_18points_val_v1.json
    IMAGE_PREFIX=${DATASET_ROOT}/BoxCars21k_crop/images/
elif [ ${TEST_SET} == "boxcars-full-val" ]; then
    DATA_TYPE=TopDownVehicleDataset
    ANN_FILE=${DATASET_ROOT}/BoxCars21k/annotations/vehicle_keypoints_18points_val_v1.json
    IMAGE_PREFIX=${DATASET_ROOT}/BoxCars21k/images/
elif [ ${TEST_SET} == "uadetrac-val" ]; then
    DATA_TYPE=TopDownVehicleDataset
    ANN_FILE=${DATASET_ROOT}/UA_DETRAC_crop_fps5/annotations/keypoints/vehicle_keypoints_18points_val_v1.json
    IMAGE_PREFIX=${DATASET_ROOT}/UA_DETRAC_crop_fps5/images/
elif [ ${TEST_SET} == "boxcars-test" ]; then
    DATA_TYPE=TopDownSimpleImageDataset
    ANN_FILE=${DATASET_ROOT}/BoxCars21k_crop/annotations/images_with_shape.list
    IMAGE_PREFIX=${DATASET_ROOT}/BoxCars21k_crop/images/
elif [ ${TEST_SET} == "uadetrac-test" ]; then
    DATA_TYPE=TopDownSimpleImageDataset
    ANN_FILE=${DATASET_ROOT}/UA_DETRAC_crop_fps5/annotations/images_with_shape.list
    IMAGE_PREFIX=${DATASET_ROOT}/UA_DETRAC_crop_fps5/images/
elif [ ${TEST_SET} == "uadetrac-tmp" ]; then
    DATA_TYPE=TopDownSimpleImageDataset
    ANN_FILE=${DATASET_ROOT}/UA_DETRAC_crop_fps5/annotations-tmp/images_with_shape.list
    IMAGE_PREFIX=${DATASET_ROOT}/UA_DETRAC_crop_fps5/tmp_images_donnot_del/
elif [ ${TEST_SET} == "bit-test" ]; then
    DATA_TYPE=TopDownSimpleImageDataset
    ANN_FILE=${DATASET_ROOT}/BITVehicle/annotations/cropped_images_with_shape.list
    IMAGE_PREFIX=${DATASET_ROOT}/BITVehicle/cropped_images/
else
    echo 'invalid test set:' ${TEST_SET}
    exit
fi

Test(){
    echo 'start testing:' ${EXP_NAME} 'config:' ${CONFIG_FILE} 'checkpoint: ' ${CHECKPOINT}
    python -u ${ROOT}/tools/test.py ${CONFIG_FILE} ${CHECKPOINT} \
        --eval mAP \
        --out ${OUTPUT_FILE} \
        --gpu-id ${GPUS} \
        --cfg-options \
        data.test.type=${DATA_TYPE} \
        data.test.ann_file=${ANN_FILE} \
        data.test.img_prefix=${IMAGE_PREFIX} \
        model.test_cfg.filter_strategy='vehicle_18points_filter' \
        2>&1 | tee logs/${JOB_NAME}.log
}

MergeGT() {

    result_boxcars=$(echo $TEST_SET | grep "boxcars")
    result_ua=$(echo $TEST_SET | grep "uadetrac")
    if [[ "$result_boxcars" != "" ]]; then
        python -u ${ROOT}/tools/dataset/merge_gt_pred_keypoints.py \
        -p ${OUTPUT_FILE} \
        -g ${DATASET_ROOT}/BoxCars21k_crop/annotations/keypoints/vehicle_keypoints_18points_train_v1.json \
            ${DATASET_ROOT}/BoxCars21k_crop/annotations/keypoints/vehicle_keypoints_18points_val_v1.json \
        --root-dir ${IMAGE_PREFIX}
    elif [[ $result_ua != "" ]]; then
        python -u ${ROOT}/tools/dataset/merge_gt_pred_keypoints.py \
        -p ${OUTPUT_FILE} \
        -g ${DATASET_ROOT}/UA_DETRAC_crop_fps5/annotations/keypoints/vehicle_keypoints_18points_train_v1.json \
            ${DATASET_ROOT}/UA_DETRAC_crop_fps5/annotations/keypoints/vehicle_keypoints_18points_val_v1.json \
        --root-dir ${IMAGE_PREFIX}
    else
        echo 'invalid test set:' ${TEST_SET} 'for merging gts'
        exit
    fi

}

Test;
# MergeGT;

echo ${JOB_NAME} 'done.'

#    --eval mAP \
# --format-only \
    # model.test_cfg.filter_strategy='vehicle_18points_filter' \
