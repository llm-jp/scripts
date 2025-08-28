# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, LLM-jp.
import json
import os
import sys
import types
import torch

from utils import print_memory_usage
from schema_core import get_model_schema

# Monkey-patch read_metadata to be CPU-safe (avoid CUDA tensors on CPU-only)
import torch.distributed as dist
from megatron.training import checkpointing as ckpt

def _read_metadata_cpu(tracker_filename):
    import sys
    import torch
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print(f'ERROR: Invalid metadata file {tracker_filename}. Exiting')
                sys.exit(1)
    assert iteration > 0 or release, f'error parsing metadata file {tracker_filename}'

    if dist.is_initialized():
        iters = torch.tensor([iteration], dtype=torch.long)  # CPU tensor
        dist.all_reduce(iters, op=dist.ReduceOp.MAX)
        max_iter = iters[0].item()
        if iteration != max_iter:
            rank = dist.get_rank()
            print(f'WARNING: on rank {rank} found iteration {iteration} in the metadata '
                  f'while max iteration across the ranks is {max_iter}, replacing it.')
    else:
        max_iter = iteration
    return max_iter, release


ckpt.read_metadata = _read_metadata_cpu


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron loader (CPU)')
    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='Original vocab size; trims padding if set.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='JSON vocab file to derive vocab size.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    group.add_argument('--position-embedding-type',
                       type=str,
                       default='learned_absolute',
                       choices=['learned_absolute', 'rope'],
                       help='Type of position embedding.')
    group.add_argument('--loader-transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')


class MegatronCheckpointLoaderBase:
    def __init__(self, args, queue, build_tokenizer=False):
        self.args = args
        self.queue = queue
        self.build_tokenizer = build_tokenizer
        self.margs = None
        self.checkpoint_args = None
        self.all_models = None
        self.md = None
        self.consumed_train_samples = None
        self.consumed_valid_samples = None

    def _maybe_parse_additional_megatron_args(self, margs, checkpoint_args):
        return margs

    def parse_megatron_args(self):
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
        if self.args.megatron_path is not None:
            sys.path.insert(0, self.args.megatron_path)
        try:
            from megatron.training.arguments import parse_args, validate_args
            from megatron.training.checkpointing import load_args_from_checkpoint
        except ModuleNotFoundError:
            print("Unable to import Megatron. Please specify --megatron-path. Exiting.")
            self.queue.put("exit")
            sys.exit(1)

        sys.argv = self.build_sys_argv()
        margs = parse_args()
        margs, checkpoint_args = load_args_from_checkpoint(margs)

        margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size
        margs.fp16 = checkpoint_args.fp16
        margs.bf16 = checkpoint_args.bf16

        if margs.expert_model_parallel_size > 1:
            margs.sequence_parallel = True

        margs = self._maybe_parse_additional_megatron_args(margs, checkpoint_args)

        try:
            from megatron.training.arguments import validate_args
            margs = validate_args(margs)
        except Exception as e:
            print(f"Error validating Megatron arguments: {e}")
            self.queue.put("exit")
            sys.exit(1)

        margs.use_legacy_models = False
        # Force CPU-safe path
        margs.transformer_impl = "local"
        # Disable all CUDA/TE/Apex fusions explicitly
        margs.apply_rope_fusion = False
        margs.bias_swiglu_fusion = False
        margs.masked_softmax_fusion = False
        margs.bias_gelu_fusion = False
        margs.bias_dropout_fusion = False
        margs.gradient_accumulation_fusion = False
        margs.gradient_reduce_div_fusion = False
        if margs.normalization == "RMSNorm":
            margs.no_persist_layer_norm = True

        self.margs = margs
        self.checkpoint_args = checkpoint_args

    def check_for_arg(self, arg_name, default=None):
        if getattr(self.margs, arg_name, None) is None:
            if default is not None:
                setattr(self.margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify argument {arg_name}. Exiting.")
                print(f"Arguments: {self.margs}")
                self.queue.put("exit")
                sys.exit(1)

    def ensure_required_arguments(self):
        self.check_for_arg('tensor_model_parallel_size')
        self.check_for_arg('pipeline_model_parallel_size')
        self.check_for_arg('num_layers')
        self.check_for_arg('hidden_size')
        self.check_for_arg('seq_length')
        self.check_for_arg('num_attention_heads')
        self.check_for_arg('max_position_embeddings')
        self.check_for_arg('position_embedding_type')
        self.check_for_arg('tokenizer_type')
        self.check_for_arg('iteration')
        self.check_for_arg('bert_binary_head')
        self.check_for_arg('params_dtype')
        self.check_for_arg('swiglu', False)

    def initialize_megatron_env(self):
        try:
            from megatron.training.global_vars import set_global_variables
            from megatron.core import mpu
        except ModuleNotFoundError as e:
            print(f"Unable to import required Megatron modules: {e}")
            self.queue.put("exit")
            sys.exit(1)

        # Init torch.distributed once
        if dist.is_available() and not dist.is_initialized():
            # Prefer env:// when under torchrun; fallback to file:// (single-rank)
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                dist.init_process_group(backend='gloo', init_method='env://')
            else:
                import tempfile
                tmp = tempfile.NamedTemporaryFile(delete=False)
                tmp.close()
                dist.init_process_group(
                    backend='gloo',
                    init_method=f'file://{tmp.name}',
                    rank=0,
                    world_size=1,
                )

        # Init model parallel groups
        try:
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.margs.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.margs.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.margs.virtual_pipeline_model_parallel_size,
                context_parallel_size=getattr(self.margs, 'context_parallel_size', 1),
                expert_model_parallel_size=self.margs.expert_model_parallel_size,
            )
        except TypeError:
            # compatible with old signature
            mpu.initialize_model_parallel(
                self.margs.tensor_model_parallel_size,
                self.margs.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.margs.virtual_pipeline_model_parallel_size,
            )

        # CPU-safe: only attempt legacy fused kernels when CUDA is available
        try:
            from torch.utils.cpp_extension import CUDA_HOME
        except Exception:
            CUDA_HOME = None
        if torch.cuda.is_available() and CUDA_HOME:
            try:
                from megatron.legacy import fused_kernels
                fused_kernels.load(self.margs)
            except Exception as e:
                print(f"Skipping fused kernels due to error: {e}")
        else:
            print("CUDA not available; skipping legacy fused kernels.")

        # Ensure num_microbatches_calculator is initialized
        from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
        init_num_microbatches_calculator(
            self.margs.rank,
            self.margs.rampup_batch_size,
            self.margs.global_batch_size,
            self.margs.micro_batch_size,
            self.margs.data_parallel_size,
            self.margs.decrease_batch_size_if_needed,
        )

    def compute_true_vocab_size(self):
        if self.args.true_vocab_size is not None:
            return self.args.true_vocab_size
        elif self.args.vocab_file is not None:
            vocab = json.load(open(self.args.vocab_file))
            return len(vocab)
        else:
            return None

    def verify_vocabs_match(self, true_vocab_size):
        if self.args.true_vocab_size is not None and self.args.vocab_file is not None:
            vocab = json.load(open(self.args.vocab_file))
            if len(vocab) != self.args.true_vocab_size:
                print("Both --true-vocab-size and --vocab-file specified but vocab sizes do not match. Aborting.")
                return False
        return True

    def load_model_shards(self, model_provider, dtype):
        from megatron.core import mpu
        try:
            import transformer_engine
        except Exception:
            sys.modules['transformer_engine'] = types.SimpleNamespace(__version__='0.0.0')
        from megatron.training.checkpointing import load_checkpoint

        consumed_train_samples = None
        consumed_valid_samples = None
        tp_size = self.margs.tensor_model_parallel_size
        pp_size = self.margs.pipeline_model_parallel_size
        vp_size = self.margs.virtual_pipeline_model_parallel_size or 1

        all_models = []

        def get_models_for_pipeline_stage(count, dtype):
            local_models_for_stage = [[] for _ in range(vp_size)]

            # Only load TP shard under multi-process; otherwise iterate all
            if dist.is_initialized() and dist.get_world_size() > 1 and mpu.get_tensor_model_parallel_world_size() > 1:
                tp_ranks = [mpu.get_tensor_model_parallel_rank()]
            else:
                tp_ranks = range(count)

            for tp_rank in tp_ranks:
                mpu.set_tensor_model_parallel_rank(tp_rank)
                model_list = []
                for i in range(vp_size):
                    mpu.set_virtual_pipeline_model_parallel_rank(i)
                    pre_process = mpu.is_pipeline_first_stage()
                    post_process = mpu.is_pipeline_last_stage()
                    this_model = model_provider(pre_process=pre_process,
                                                post_process=post_process).to(dtype)
                    model_list.append(this_model)

                # Ensure counters are zeroed and load
                self.margs.consumed_train_samples = 0
                self.margs.consumed_valid_samples = 0
                self.margs.exit_on_missing_checkpoint = True
                load_checkpoint(model_list, None, None)

                nonlocal consumed_train_samples, consumed_valid_samples
                if consumed_train_samples is not None:
                    assert self.margs.consumed_train_samples == consumed_train_samples
                else:
                    consumed_train_samples = self.margs.consumed_train_samples
                if consumed_valid_samples is not None:
                    assert self.margs.consumed_valid_samples == consumed_valid_samples
                else:
                    consumed_valid_samples = self.margs.consumed_valid_samples

                for vp_rank in range(vp_size):
                    local_models_for_stage[vp_rank].append(model_list[vp_rank])

                print_memory_usage("loader", tp_rank, count)

            return local_models_for_stage

        mpu.set_virtual_pipeline_model_parallel_rank(0)
        for pp_rank in range(pp_size):
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            all_models.append(get_models_for_pipeline_stage(tp_size, dtype))

        return all_models, consumed_train_samples, consumed_valid_samples

    def queue_put(self, name, msg):
        print(f"sending {name}")
        msg["name"] = name
        self.queue.put(msg)

    def _tp_gather_cat_rank0(self, t: torch.Tensor, dim: int):
        from megatron.core import mpu
        if not (dist.is_initialized() and mpu.get_tensor_model_parallel_world_size() > 1):
            return t
        group = mpu.get_tensor_model_parallel_group()
        ws = mpu.get_tensor_model_parallel_world_size()
        parts = [torch.empty_like(t) for _ in range(ws)]
        dist.all_gather(parts, t, group=group)
        if mpu.get_tensor_model_parallel_rank() == 0:
            return torch.cat(parts, dim=dim)
        else:
            return None

    def send_llm_over_queue(self, schema):
        from megatron.core import mpu

        tp_size = self.margs.tensor_model_parallel_size
        pp_size = self.margs.pipeline_model_parallel_size
        vp_size = self.margs.virtual_pipeline_model_parallel_size or 1

        # Only main TP rank sends to saver; other ranks only participate in communication
        is_main_tp = (not dist.is_initialized()) or \
                     (mpu.get_tensor_model_parallel_world_size() == 1) or \
                     (mpu.get_tensor_model_parallel_rank() == 0)

        first_pipeline_models = self.all_models[0][0]

        # Embeddings
        embeddings = [schema.get("embeddings", m) for m in first_pipeline_models]
        message = {
            "word embeddings": torch.cat([e["word"] for e in embeddings], dim=0)
        }
        # Gather full embeddings across TP
        message["word embeddings"] = self._tp_gather_cat_rank0(message["word embeddings"], dim=0)

        if self.md.position_embedding_type == 'learned_absolute':
            message["position embeddings"] = embeddings[0]["pos"]
        else:
            assert embeddings[0]["pos"] is None
        if is_main_tp:
            self.queue_put("embeddings", message)

        total_layer_num = 0
        for vp_rank in range(vp_size):
            for pp_rank in range(pp_size):
                models = self.all_models[pp_rank][vp_rank]
                num_layers = schema.get_num_layers(models[0])
                for layer_idx in range(num_layers):
                    message = {}
                    layer = schema.get_layer(models[0], layer_idx)

                    message["input norm weight"] = layer["self_attn_norm_weight"]
                    message["post norm weight"] = layer["mlp_norm_weight"]
                    if self.md.norm_has_bias:
                        message["input norm bias"] = layer["self_attn_norm_bias"]
                        message["post norm bias"] = layer["mlp_norm_bias"]
                    if self.md.linear_bias:
                        message["dense bias"] = layer["self_attn_proj_bias"]
                        message["mlp l1 bias"] = layer["mlp_fc2_bias"]

                    qkv_weight, qkv_bias = [], []
                    dense_weight = []
                    mlp_l0_weight, mlp_l0_bias = [], []
                    mlp_l1_weight = []

                    # collect local TP-shard tensors (on this rank)
                    for model_tp in models:
                        layer_p = schema.get_layer(model_tp, layer_idx)
                        qkv_weight.append(layer_p["self_attn_qkv_weight"])
                        dense_weight.append(layer_p["self_attn_proj_weight"])
                        mlp_l0_weight.append(layer_p["mlp_fc1_weight"])
                        mlp_l1_weight.append(layer_p["mlp_fc2_weight"])
                        if self.md.qkv_bias:
                            qkv_bias.append(layer_p["self_attn_qkv_bias"])
                        if self.md.linear_bias:
                            mlp_l0_bias.append(layer_p["mlp_fc1_bias"])

                    # Build message with local concat
                    if self.md.swiglu:
                        for i in range(len(mlp_l0_weight)):
                            mlp_l0_weight[i] = torch.chunk(mlp_l0_weight[i], 2, dim=0)
                        message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
                        message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
                    else:
                        message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

                    message["qkv weight"] = torch.cat(qkv_weight, dim=0)
                    message["dense weight"] = torch.cat(dense_weight, dim=1)
                    message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)

                    if self.md.qkv_bias:
                        message["qkv bias"] = torch.cat(qkv_bias, dim=0)
                    if self.md.linear_bias:
                        if self.md.swiglu:
                            for i in range(len(mlp_l0_bias)):
                                mlp_l0_bias[i] = torch.chunk(mlp_l0_bias[i], 2, dim=0)
                            message["mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias], dim=0)
                            message["mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias], dim=0)
                        else:
                            message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)

                    # Now gather to full tensors across TP
                    qkv_w  = self._tp_gather_cat_rank0(message["qkv weight"],  dim=0)
                    dense  = self._tp_gather_cat_rank0(message["dense weight"], dim=1)
                    mlp_l1 = self._tp_gather_cat_rank0(message["mlp l1 weight"], dim=1)
                    if self.md.qkv_bias:
                        qkv_b = self._tp_gather_cat_rank0(message["qkv bias"], dim=0)
                    if self.md.swiglu:
                        mlp0W = self._tp_gather_cat_rank0(message["mlp l0 weight W"], dim=0)
                        mlp0V = self._tp_gather_cat_rank0(message["mlp l0 weight V"], dim=0)
                        if self.md.linear_bias:
                            mlp0bW = self._tp_gather_cat_rank0(message["mlp l0 bias W"], dim=0)
                            mlp0bV = self._tp_gather_cat_rank0(message["mlp l0 bias V"], dim=0)
                    else:
                        mlp0 = self._tp_gather_cat_rank0(message["mlp l0 weight"], dim=0)
                        if self.md.linear_bias:
                            mlp0b = self._tp_gather_cat_rank0(message["mlp l0 bias"], dim=0)

                    if is_main_tp:
                        message["qkv weight"]   = qkv_w
                        message["dense weight"] = dense
                        message["mlp l1 weight"]= mlp_l1
                        if self.md.qkv_bias: message["qkv bias"] = qkv_b
                        if self.md.swiglu:
                            message["mlp l0 weight W"] = mlp0W
                            message["mlp l0 weight V"] = mlp0V
                            if self.md.linear_bias:
                                message["mlp l0 bias W"] = mlp0bW
                                message["mlp l0 bias V"] = mlp0bV
                        else:
                            message["mlp l0 weight"] = mlp0
                            if self.md.linear_bias: message["mlp l0 bias"] = mlp0b

                        self.queue_put(f"transformer layer {total_layer_num}", message)
                    total_layer_num += 1

        # Final norm
        models = self.all_models[0][0]
        final_norm = schema.get("final_norm", models[0])
        if is_main_tp:
            self.queue_put("final norm", {"weight": final_norm["weight"], **({"bias": final_norm["bias"]} if self.md.norm_has_bias else {})})

        # Output layer
        if self.md.output_layer:
            output_layers = [schema.get("output_layer", m) for m in models]
            message = {
                "weight": torch.cat([layer["weight"] for layer in output_layers], dim=0),
            }
            w_all = self._tp_gather_cat_rank0(message["weight"], dim=0)
            if is_main_tp:
                message["weight"] = w_all
                self.queue_put("output layer", message)

    def build_checkpoint_metadata(self, true_vocab_size):
        norm_has_bias = True
        if hasattr(self.checkpoint_args, 'normalization'):
            norm_has_bias = (self.checkpoint_args.normalization == "LayerNorm")
        md = types.SimpleNamespace()
        md.model_type = self.args.model_type
        md.num_layers = self.margs.num_layers
        md.hidden_size = self.margs.hidden_size
        md.seq_length = self.margs.seq_length
        md.num_attention_heads = self.margs.num_attention_heads
        md.max_position_embeddings = self.margs.max_position_embeddings
        md.tokenizer_type = self.margs.tokenizer_type
        md.iteration = self.margs.iteration
        md.params_dtype = self.margs.params_dtype
        md.bert_binary_head = self.margs.bert_binary_head
        md.output_layer = self.margs.untie_embeddings_and_output_weights
        md.position_embedding_type = self.margs.position_embedding_type
        md.linear_bias = self.margs.add_bias_linear
        md.qkv_bias = self.margs.add_qkv_bias
        md.norm_has_bias = norm_has_bias
        md.swiglu = self.margs.swiglu
        md.previous_tensor_parallel_size = self.margs.tensor_model_parallel_size
        md.previous_pipeline_parallel_size = self.margs.pipeline_model_parallel_size
        md.true_vocab_size = true_vocab_size
        md.make_vocab_size_divisible_by = self.margs.make_vocab_size_divisible_by
        md.checkpoint_args = self.checkpoint_args
        md.use_legacy_models = self.margs.use_legacy_models
        return md

    def build_sys_argv(self):
        return [
            'script.py',
            '--no-masked-softmax-fusion',
            '--no-bias-gelu-fusion',
            '--no-bias-dropout-fusion',
            '--no-bias-swiglu-fusion',
            '--no-rope-fusion',
            '--no-async-tensor-model-parallel-allreduce',
            '--use-cpu-initialization',
            '--micro-batch-size', '1',
            '--no-load-optim',
            '--no-load-rng',
            '--no-save-optim',
            '--no-save-rng',
            '--no-initialization',
            '--mock-data',
            '--load', self.args.load_dir,
            '--exit-on-missing-checkpoint',
            '--use-mp-args-from-checkpoint-args',
            '--no-one-logger',
            '--distributed-backend', 'gloo',
            '--attention-backend', 'unfused',
            '--transformer-impl', 'local',
            '--no-gradient-accumulation-fusion',
            '--no-gradient-reduce-div-fusion',
            '--tp-comm-bootstrap-backend', 'gloo',
            '--finetune',
        ]

    def import_model_provider(self):
        raise NotImplementedError

    def load(self):
        self.parse_megatron_args()
        self.ensure_required_arguments()

        from megatron.training.global_vars import set_args as _set_args
        _set_args(self.margs)

        self.initialize_megatron_env()

        model_provider = self.import_model_provider()

        # True vocab verification
        true_vocab_size = self.compute_true_vocab_size()
        if not self.verify_vocabs_match(true_vocab_size):
            self.queue.put("exit")
            sys.exit(1)

        # Build metadata
        self.md = self.build_checkpoint_metadata(true_vocab_size)

        # Load model shards
        self.all_models, self.consumed_train_samples, self.consumed_valid_samples = self.load_model_shards(
            model_provider, self.md.params_dtype
        )

        self.send_model_over_queue()

    def send_model_over_queue(self):
        from megatron.core import mpu
        is_main_tp = (not dist.is_initialized()) or (mpu.get_tensor_model_parallel_world_size() == 1) or (mpu.get_tensor_model_parallel_rank() == 0)

        self.send_metadata_over_queue()
        schema = get_model_schema(
            self.md.model_type,
            self.margs.transformer_impl,
            self.margs.num_experts,
            self.margs.expert_model_parallel_size,
        )
        self.send_llm_over_queue(schema)

        if is_main_tp:
            self.queue.put("done")
        else:
            self.queue.put("exit")

    def send_metadata_over_queue(self):
        self.md.consumed_train_samples = self.consumed_train_samples
        self.md.consumed_valid_samples = self.consumed_valid_samples
        self.queue.put(self.md)


class MegatronCheckpointLoaderLLM(MegatronCheckpointLoaderBase):
    def build_sys_argv(self):
        return [
            *super().build_sys_argv(),
            '--position-embedding-type', self.args.position_embedding_type,
        ]

    def import_model_provider(self):
        if self.args.model_type == 'GPT':
            from pretrain_gpt import model_provider
            return model_provider
        elif self.args.model_type == 'BERT':
            from pretrain_bert import model_provider
            return model_provider
        else:
            raise Exception(f"Unrecognized model type: {self.args.model_type}")

    def send_model_over_queue(self):
        from megatron.core import mpu
        is_main_tp = (not dist.is_initialized()) or (mpu.get_tensor_model_parallel_world_size() == 1) or (mpu.get_tensor_model_parallel_rank() == 0)

        self.send_metadata_over_queue()
        schema = get_model_schema(
            self.md.model_type,
            self.margs.transformer_impl,
            self.margs.num_experts,
            self.margs.expert_model_parallel_size,
        )
        self.send_llm_over_queue(schema)

        if is_main_tp:
            self.queue.put("done")
        else:
            self.queue.put("exit")

    def send_metadata_over_queue(self):
        self.md.consumed_train_samples = self.consumed_train_samples
        self.md.consumed_valid_samples = self.consumed_valid_samples
        self.queue.put(self.md)


def load_checkpoint(queue, args):
    loader = MegatronCheckpointLoaderLLM(args, queue)
    try:
        loader.load()
    except Exception as e:
        queue.put("exit")
        raise e
