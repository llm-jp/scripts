#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -N 0190_pretrain
#PBS -v RTYPE=rt_HF
#PBS -l select=32:ncpus=8:mpiprocs=8:ngpus=8
#PBS -l walltime=168:00:00
#PBS -j oe
#PBS -koed
#PBS -V
#PBS -o outputs/exp3
#PBS -m n

cd $PBS_O_WORKDIR

EXP_NAME="exp3"

set -eu -o pipefail

# Setup environment
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
module load cudnn/9.5/9.5.1
module load nccl/2.25/2.25.1-1
module load hpcx/2.20
source venv/bin/activate

JOB_ID=$(echo $PBS_JOBID | cut -d. -f1)
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE | hostname -f)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

NUM_NODES=$(sort -u $PBS_NODEFILE | wc -l)
NUM_GPUS_PER_NODE=8

# Debug/logging flags
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
HIDDEN_SIZE=2560
FFN_HIDDEN_SIZE=960
MOE_FFN_HIDDEN_SIZE=960
NUM_LAYERS=32
NUM_HEADS=40
NUM_QUERY_GROUPS=4
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
EXPERT_PARALLEL_SIZE=4
CONTEXT_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE} * ${EXPERT_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=1024

LR=4e-4
MIN_LR=4e-5
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# total number of iterations
LR_WARMUP_STEPS=2000
LR_DECAY_ITERS=100000
TRAIN_STEPS=100000

CACHE_DIR=cache

# model config
TOKENIZER_MODEL="src/llm-jp-tokenizer-v4/v4_alpha_1.0.model"
CHECKPOINT_LOAD_DIR=checkpoints/megatron/${EXP_NAME}
CHECKPOINT_SAVE_DIR=checkpoints/megatron/${EXP_NAME}

mkdir -p ${CHECKPOINT_SAVE_DIR}
DATASET_ROOT="/groups/gcg51557/experiments/0212_v4-train-data/data/v20250816/tokenized"

TRAIN_DATA_PATH=""
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 855056046544 ${DATASET_ROOT}/code_olmo-starcoder_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 942818994776 ${DATASET_ROOT}/code_stack_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 41119172160 ${DATASET_ROOT}/en_dolma-books_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 480292134384 ${DATASET_ROOT}/en_dolma-pes2o_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 658034367288 ${DATASET_ROOT}/en_dolma-reddit_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 30860169664 ${DATASET_ROOT}/en_dolma-wiki_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 11867358448 ${DATASET_ROOT}/en_dolmino-stackexchange_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 25134216 ${DATASET_ROOT}/en_gsm8k_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 70000284832 ${DATASET_ROOT}/en_mathpile_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 103817401008 ${DATASET_ROOT}/en_olmo-algebraicstack_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 173730424536 ${DATASET_ROOT}/en_olmo-arxiv_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 105368433136 ${DATASET_ROOT}/en_olmo-openwebmath_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 37973097112 ${DATASET_ROOT}/en_wiki_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 102568520111 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 102509087783 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100816401574 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0002_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100065810915 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0003_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 99955083033 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0004_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100268985585 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0005_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 98582941635 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0006_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 102501814684 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0007_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 87146714512 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0008_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 102540352684 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 101615714055 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100223579527 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0002_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 99832954483 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0003_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100200285660 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0004_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 98939237258 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0005_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 102361066324 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0006_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100127948990 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0007_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 51572633747 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0008_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 102346538767 ${DATASET_ROOT}/en_fineweb-rescored_score_12_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100658650543 ${DATASET_ROOT}/en_fineweb-rescored_score_12_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 99879930853 ${DATASET_ROOT}/en_fineweb-rescored_score_12_0002_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100207021051 ${DATASET_ROOT}/en_fineweb-rescored_score_12_0003_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 99609557924 ${DATASET_ROOT}/en_fineweb-rescored_score_12_0004_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 101835220299 ${DATASET_ROOT}/en_fineweb-rescored_score_12_0005_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 99887438413 ${DATASET_ROOT}/en_fineweb-rescored_score_12_0006_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 91670119172 ${DATASET_ROOT}/en_fineweb-rescored_score_12_0007_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 101899332058 ${DATASET_ROOT}/en_fineweb-rescored_score_13_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100107151548 ${DATASET_ROOT}/en_fineweb-rescored_score_13_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100188263808 ${DATASET_ROOT}/en_fineweb-rescored_score_13_0002_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100208220130 ${DATASET_ROOT}/en_fineweb-rescored_score_13_0003_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 101460435434 ${DATASET_ROOT}/en_fineweb-rescored_score_13_0004_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 99804308223 ${DATASET_ROOT}/en_fineweb-rescored_score_13_0005_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 99868561720 ${DATASET_ROOT}/en_fineweb-rescored_score_13_0006_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 28118083173 ${DATASET_ROOT}/en_fineweb-rescored_score_13_0007_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 101287815642 ${DATASET_ROOT}/en_fineweb-rescored_score_14_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100193448611 ${DATASET_ROOT}/en_fineweb-rescored_score_14_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100909877098 ${DATASET_ROOT}/en_fineweb-rescored_score_14_0002_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100757143238 ${DATASET_ROOT}/en_fineweb-rescored_score_14_0003_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 99723340081 ${DATASET_ROOT}/en_fineweb-rescored_score_14_0004_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 99265358219 ${DATASET_ROOT}/en_fineweb-rescored_score_14_0005_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 15921206946 ${DATASET_ROOT}/en_fineweb-rescored_score_14_0006_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 101005877270 ${DATASET_ROOT}/en_fineweb-rescored_score_15_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100489515421 ${DATASET_ROOT}/en_fineweb-rescored_score_15_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 101723685894 ${DATASET_ROOT}/en_fineweb-rescored_score_15_0002_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100045434530 ${DATASET_ROOT}/en_fineweb-rescored_score_15_0003_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 99720256993 ${DATASET_ROOT}/en_fineweb-rescored_score_15_0004_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 98543462382 ${DATASET_ROOT}/en_fineweb-rescored_score_15_0005_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 7338842460 ${DATASET_ROOT}/en_fineweb-rescored_score_15_0006_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100825885054 ${DATASET_ROOT}/en_fineweb-rescored_score_16_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 101739112228 ${DATASET_ROOT}/en_fineweb-rescored_score_16_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100416142859 ${DATASET_ROOT}/en_fineweb-rescored_score_16_0002_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 99754212843 ${DATASET_ROOT}/en_fineweb-rescored_score_16_0003_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100042039542 ${DATASET_ROOT}/en_fineweb-rescored_score_16_0004_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 45275202852 ${DATASET_ROOT}/en_fineweb-rescored_score_16_0005_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 101199070911 ${DATASET_ROOT}/en_fineweb-rescored_score_17_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 101438063034 ${DATASET_ROOT}/en_fineweb-rescored_score_17_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 99938292420 ${DATASET_ROOT}/en_fineweb-rescored_score_17_0002_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 99892259073 ${DATASET_ROOT}/en_fineweb-rescored_score_17_0003_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 88320720228 ${DATASET_ROOT}/en_fineweb-rescored_score_17_0004_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 101682793329 ${DATASET_ROOT}/en_fineweb-rescored_score_18_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100779083985 ${DATASET_ROOT}/en_fineweb-rescored_score_18_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 99807036301 ${DATASET_ROOT}/en_fineweb-rescored_score_18_0002_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100032616702 ${DATASET_ROOT}/en_fineweb-rescored_score_18_0003_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 34415854865 ${DATASET_ROOT}/en_fineweb-rescored_score_18_0004_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 101794562305 ${DATASET_ROOT}/en_fineweb-rescored_score_19_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100159790313 ${DATASET_ROOT}/en_fineweb-rescored_score_19_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 100003883373 ${DATASET_ROOT}/en_fineweb-rescored_score_19_0002_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 56293984847 ${DATASET_ROOT}/en_fineweb-rescored_score_19_0003_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 407638433572 ${DATASET_ROOT}/en_fineweb-rescored_score_20_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 401007648012 ${DATASET_ROOT}/en_fineweb-rescored_score_20_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 400098710148 ${DATASET_ROOT}/en_fineweb-rescored_score_20_0002_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 310471544996 ${DATASET_ROOT}/en_fineweb-rescored_score_20_0003_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 405979851888 ${DATASET_ROOT}/en_fineweb-rescored_score_21_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 399957905920 ${DATASET_ROOT}/en_fineweb-rescored_score_21_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 290440168380 ${DATASET_ROOT}/en_fineweb-rescored_score_21_0002_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 405698461528 ${DATASET_ROOT}/en_fineweb-rescored_score_22_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 399908643392 ${DATASET_ROOT}/en_fineweb-rescored_score_22_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 304268233784 ${DATASET_ROOT}/en_fineweb-rescored_score_22_0002_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 403950503136 ${DATASET_ROOT}/en_fineweb-rescored_score_23_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 397091572144 ${DATASET_ROOT}/en_fineweb-rescored_score_23_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17630260124 ${DATASET_ROOT}/en_fineweb-rescored_score_23_0002_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 403062653976 ${DATASET_ROOT}/en_fineweb-rescored_score_24_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 303389196312 ${DATASET_ROOT}/en_fineweb-rescored_score_24_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 402915428660 ${DATASET_ROOT}/en_fineweb-rescored_score_25_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 293582191580 ${DATASET_ROOT}/en_fineweb-rescored_score_25_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 401842757612 ${DATASET_ROOT}/en_fineweb-rescored_score_26_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 95148322400 ${DATASET_ROOT}/en_fineweb-rescored_score_26_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 401610618036 ${DATASET_ROOT}/en_fineweb-rescored_score_27_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 73154930544 ${DATASET_ROOT}/en_fineweb-rescored_score_27_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 326746055548 ${DATASET_ROOT}/en_fineweb-rescored_score_28_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 260583674604 ${DATASET_ROOT}/en_fineweb-rescored_score_29_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 401194947784 ${DATASET_ROOT}/en_fineweb-rescored_score_30_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 270538421040 ${DATASET_ROOT}/en_fineweb-rescored_score_30_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 996302704 ${DATASET_ROOT}/ja_aozorabunko_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 99809039432 ${DATASET_ROOT}/ja_ceek-news_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 541520712 ${DATASET_ROOT}/ja_e-gov_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 6179435824 ${DATASET_ROOT}/ja_kaken_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5387944368 ${DATASET_ROOT}/ja_kokkai-giji_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 130044244728 ${DATASET_ROOT}/ja_nwc2010_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 206902590720 ${DATASET_ROOT}/ja_nwjc_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 486510753720 ${DATASET_ROOT}/ja_patent_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 90962164248 ${DATASET_ROOT}/ja_sip-comprehensive-html_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 226818645136 ${DATASET_ROOT}/ja_sip-comprehensive-pdf-pdf2text_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5930050328 ${DATASET_ROOT}/ja_warp-html_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 76509752040 ${DATASET_ROOT}/ja_warp-pdf-e0_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 343134486568 ${DATASET_ROOT}/ja_warp-pdf-e0.2_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8681002704 ${DATASET_ROOT}/ja_wiki_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 198918889396 ${DATASET_ROOT}/ja_cc_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 197477284040 ${DATASET_ROOT}/ja_cc_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 198629681700 ${DATASET_ROOT}/ja_cc_0002_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 201315333292 ${DATASET_ROOT}/ja_cc_0003_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 73316218724 ${DATASET_ROOT}/ja_cc_0004_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 168717734020 ${DATASET_ROOT}/ja_fineweb-2_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 170947462036 ${DATASET_ROOT}/ja_fineweb-2_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 169864760144 ${DATASET_ROOT}/ja_fineweb-2_0002_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 169663322804 ${DATASET_ROOT}/ja_fineweb-2_0003_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 168161765892 ${DATASET_ROOT}/ja_fineweb-2_0004_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17023262332 ${DATASET_ROOT}/ja_fineweb-2_0005_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2816594432 ${DATASET_ROOT}/ko_wiki_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 48038910925 ${DATASET_ROOT}/ko_fineweb2_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 5926039312 ${DATASET_ROOT}/zh_wiki_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 136502282670 ${DATASET_ROOT}/zh_fineweb2_0000_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 135056311908 ${DATASET_ROOT}/zh_fineweb2_0001_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 138369517441 ${DATASET_ROOT}/zh_fineweb2_0002_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 145115884006 ${DATASET_ROOT}/zh_fineweb2_0003_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 11414468604 ${DATASET_ROOT}/zh_fineweb2_0004_text_document"

WANDB_ENTITY="llm-jp"
WANDB_PROJECT="0190_moe"
WANDB_NAME="${EXP_NAME}"

# Model arguments
MODEL_ARGS=(
    --attention-dropout 0.0
    --disable-bias-linear
    --ffn-hidden-size ${FFN_HIDDEN_SIZE}
    --group-query-attention
    --hidden-dropout 0.0
    --hidden-size ${HIDDEN_SIZE}
    --init-method-std 0.02
    --kv-channels 128
    --max-position-embeddings ${SEQ_LENGTH}
    --moe-ffn-hidden-size ${MOE_FFN_HIDDEN_SIZE}
    --no-masked-softmax-fusion
    --norm-epsilon 1e-6
    --normalization RMSNorm
    --num-attention-heads ${NUM_HEADS}
    --num-layers ${NUM_LAYERS}
    --num-query-groups ${NUM_QUERY_GROUPS}
    --position-embedding-type rope
    --qk-layernorm
    --rotary-base 500000
    --seq-length ${SEQ_LENGTH}
    --swiglu
    --untie-embeddings-and-output-weights
    --use-mcore-models
)

MOE_ARGS=(
    --manual-gc
    --manual-gc-interval 5
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-dtype fp32
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8
    --moe-token-dispatcher-type alltoall
    --moe-z-loss-coeff 1e-3
    --num-experts 128
    --overlap-grad-reduce
    --overlap-param-gather
    #    --recompute-granularity selective
    #    --recompute-modules moe_act layernorm
)

DATA_ARGS=(
    --data-cache-path $CACHE_DIR
    --data-path $TRAIN_DATA_PATH
    --split 1,0,0
    --tokenizer-model ${TOKENIZER_MODEL}
    --tokenizer-type Llama2Tokenizer
)

TRAINING_ARGS=(
    --accumulate-allreduce-grads-in-fp32
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --async-save
    --attention-backend flash
    --attention-softmax-in-fp32
    --bf16
    --ckpt-format torch_dist
    --clip-grad ${GRAD_CLIP}
    --cross-entropy-fusion-impl native
    --cross-entropy-loss-fusion
    --distributed-backend nccl
    --distributed-timeout-minutes 120
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --lr ${LR}
    --lr-decay-iters ${TRAIN_STEPS}
    --lr-decay-style cosine
    --lr-warmup-iters ${LR_WARMUP_STEPS}
    --micro-batch-size ${MICRO_BATCH_SIZE}
    --min-lr ${MIN_LR}
    --num-workers 8
    --optimizer adam
    --train-iters ${TRAIN_STEPS}
    --transformer-impl "transformer_engine"
    --use-flash-attn
    --weight-decay ${WEIGHT_DECAY}
)

# Model parameters
MODEL_PARALLEL_ARGS=(
    --context-parallel-size ${CONTEXT_PARALLEL_SIZE}
    --expert-model-parallel-size ${EXPERT_PARALLEL_SIZE}
    --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE}
    --sequence-parallel
    --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE}
    --use-distributed-optimizer
)

LOGGING_ARGS=(
    --eval-interval 999999999
    --eval-iters 0
    --load ${CHECKPOINT_LOAD_DIR}
    --log-interval 1
    --log-throughput
    --moe-per-layer-logging
    --save ${CHECKPOINT_SAVE_DIR}
    --save-interval 1000
    --use-mpi
    --wandb-entity ${WANDB_ENTITY}
    --wandb-exp-name ${WANDB_NAME}
    --wandb-project ${WANDB_PROJECT}
)

mpirun \
    -np $NUM_GPUS \
    --hostfile $PBS_NODEFILE \
    --npernode $NUM_GPUS_PER_NODE \
    --map-by slot \
    --bind-to none \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -x NUM_NODES=$NUM_NODES \
    -x NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE \
    -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -x PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    -x PATH \
    -x LD_LIBRARY_PATH \
    python src/Megatron-LM/pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}"
