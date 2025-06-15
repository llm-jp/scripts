#!/bin/bash

set -eu -o pipefail

source scripts/environment.sh
source scripts/mpi_variables.sh
source venv/bin/activate

# open file limit
ulimit -n 65536 1048576

export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_LOGDEST_DBG=stderr
export CUDNN_LOGERR_DBG=1

NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))

# model config
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=1792
NUM_LAYERS=24
NUM_HEADS=16
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=2
EXPERT_PARALLEL_SIZE=8
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE} * ${EXPERT_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=1024

LR=1.2e-3
MIN_LR=1.2e-4
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# total number of iterations
# 2072488058295 (number of tokens) / 4096 (seq len) / 1024 (batch size) = 494119.65806365 -> 494120
LR_WARMUP_STEPS=2000
LR_DECAY_ITERS=492120
TRAIN_STEPS=492120

CACHE_DIR=cache_v4_exp1

# model config
TOKENIZER_MODEL=src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model
CHECKPOINT_LOAD_DIR=checkpoints/megatron/16expert_4granularity_0shared_top8_17b_active2.9b_z_loss_lr_1.2e-3_v4_exp1
CHECKPOINT_SAVE_DIR=checkpoints/megatron/16expert_4granularity_0shared_top8_17b_active2.9b_z_loss_lr_1.2e-3_v4_exp1

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

TRAIN_DATA_PATH=""

# code
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 104427769064 /home/shared/experiments/0111_v4-setup/corpus/tokenized/code/code_olmo-starcoder_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 114051163723 /home/shared/experiments/0111_v4-setup/corpus/tokenized/code/code_stack_0000_text_document"

# en
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5494262694 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_dolma-books_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 62853772802 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_dolma-pes2o_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 83015186637 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_dolma-reddit_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3896965449 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_dolma-wiki_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1464772187 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_dolmino-stackexchange_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 10335599308 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_finemath-4plus_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4490282853 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2013-20_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4466884009 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2013-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4537163180 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2014-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4224799151 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2014-15_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4860415280 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2014-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4536509376 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2014-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4694256938 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2014-41_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4332993557 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2014-42_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3800854267 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2014-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4548282477 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2014-52_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4181235817 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2015-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4270806510 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2015-11_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4025416201 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2015-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4586674479 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2015-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4524610853 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2015-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4029188495 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2015-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4206129891 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2015-32_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4285136588 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2015-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3423188080 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2015-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4238029622 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2015-48_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4136881894 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2016-07_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3611887150 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2016-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3776035965 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2016-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3126909842 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2016-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4223949678 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2016-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4146809288 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2016-36_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4676081324 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2016-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5451556492 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2016-44_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5310003355 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2016-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5611139438 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2017-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5796541235 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2017-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6866843713 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2017-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6932845158 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2017-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5069630293 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2017-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5482240565 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2017-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5099538436 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2017-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5378574298 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2017-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5053120049 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2017-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5908518717 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2017-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5238077875 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2017-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4794379721 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2017-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5653596841 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2018-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5454160663 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2018-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5219394061 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2018-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4914242046 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2018-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4400054233 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2018-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5380635332 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2018-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5557289601 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2018-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4681002215 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2018-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4954403474 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2018-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5399527240 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2018-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5023750210 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2018-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5631695750 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2018-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5024961497 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2019-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5527929958 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2019-09_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5069074743 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2019-13_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5135276891 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2019-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5163483537 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2019-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4929145601 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2019-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4945417546 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2019-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5423061037 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2019-35_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4767582856 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2019-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5072130421 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2019-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4772506831 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2019-47_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4359902673 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2019-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5947536081 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2020-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4492821700 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2020-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5601325808 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2020-16_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4789951627 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2020-24_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6058383287 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2020-29_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4723175465 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2020-34_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6606422623 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2020-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5164369535 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2020-45_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5062470749 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2020-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6808535394 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2021-04_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5607396258 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2021-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6860812696 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2021-17_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5623765054 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2021-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5069097528 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2021-25_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7305728374 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2021-31_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6429413016 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2021-39_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7449313770 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2021-43_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5000475286 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2021-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6405696533 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2022-05_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8080940335 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2022-21_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7276801917 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2022-27_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5324414049 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2022-33_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7620778932 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2022-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8013850304 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2022-49_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7915624626 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2023-06_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7878543407 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2023-14_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8443821164 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2023-23_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8916616626 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2023-40_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8607366889 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2023-50_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5652957183 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2024-10_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5477000460 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2024-18_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5278937671 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2024-22_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4807097186 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2024-26_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4701742260 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2024-30_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4027210453 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2024-33_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4911653572 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2024-38_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4096673829 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2024-42_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4487135744 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2024-46_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4701389300 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_fineweb-eduscore2_CC-MAIN-2024-51_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2781710 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_gsm8k_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 9176535715 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_mathpile_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 13280211413 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_olmo-algebraicstack_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 22219529548 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_olmo-arxiv_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 13395295861 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_olmo-openwebmath_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4744259830 /home/shared/experiments/0111_v4-setup/corpus/tokenized/en/en_wiki_0000_text_document"

WANDB_ENTITY="llm-jp"
WANDB_PROJECT="v3-8x1.8b"
WANDB_NAME="16expert_4granularity_0shared_top8_17b_active2.9b_z_loss_lr_1.2e-3_v4_exp1"


# Model arguments
MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length ${SEQ_LENGTH}
    --max-position-embeddings ${SEQ_LENGTH}
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --ffn-hidden-size ${FFN_HIDDEN_SIZE}
    --num-attention-heads ${NUM_HEADS}
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --num-query-groups 16
    --no-masked-softmax-fusion
    --rotary-base 10000
)

MOE_ARGS=(
    --num-experts 64
    --moe-router-topk 8
    --moe-z-loss-coeff 1e-3
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $TRAIN_DATA_PATH
    --data-cache-path $CACHE_DIR
    --split 1,0,0
)

TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --lr ${LR}
    --train-iters ${TRAIN_STEPS}
    --lr-decay-iters ${TRAIN_STEPS}
    --lr-decay-style cosine
    --min-lr ${MIN_LR}
    --weight-decay ${WEIGHT_DECAY}
    --lr-warmup-iters ${LR_WARMUP_STEPS}
    --clip-grad ${GRAD_CLIP}
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --bf16
    --use-flash-attn
    --transformer-impl "transformer_engine"
    --attention-softmax-in-fp32
    --accumulate-allreduce-grads-in-fp32
    --distributed-backend nccl
    --ckpt-format torch
)

# Model parameters
MODEL_PARALLEL_ARGS=(
   --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE}
   --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE}
   --expert-model-parallel-size ${EXPERT_PARALLEL_SIZE}
   --context-parallel-size ${CONTEXT_PARALLEL_SIZE}
   --use-distributed-optimizer
   --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --moe-per-layer-logging
    --save-interval 500
    --eval-interval ${TRAIN_STEPS}
    --eval-iters 0
    --save ${CHECKPOINT_SAVE_DIR}
    --load ${CHECKPOINT_LOAD_DIR}
    --use-mpi
    --wandb-project ${WANDB_PROJECT}
    --wandb-exp-name ${WANDB_NAME}
    --wandb-entity ${WANDB_ENTITY}
)


export NVTE_FUSED_ATTN=0
python src/Megatron-LM/pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}"
