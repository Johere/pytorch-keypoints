#!/usr/bin/env bash
set -x

ROOT=..
export PYTHONPATH=${ROOT}:${PYTHONPATH}


CONFIG_FILE=${1-${ROOT}/configs/vehicle/2d_kpt_sview_rgb_img/topdown_heatmap/hourglass52_boxcars_256x256.py}

INPUT_HEIGHT=${2-256}
INPUT_WIDTH=${3-${INPUT_HEIGHT}}

echo 'model complexity esitimate, config:' ${CONFIG_FILE}
python ${ROOT}/tools/analysis/get_flops.py \
        ${CONFIG_FILE} \
        --shape ${INPUT_HEIGHT} ${INPUT_WIDTH}
        
# ```shell
# python tools/get_flops/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
# ```

# 用户将获得如下结果：

# ```
# ==============================
# Input shape: (3, 224, 224)
# Flops: 4.12 GFLOPs
# Params: 25.56 M
# ==============================
# ```