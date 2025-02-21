import glob
import os
import shutil
from typing import Optional

import torch.multiprocessing as mp
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.get_rank import is_global_rank_zero
from nemo_aligner.algorithms.supervised import SupervisedTrainer
from nemo_aligner.models.nlp.gpt.gpt_sft_model import GPTSFTModel
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_peft,
    init_using_ptl,
    resolve_and_create_trainer,
    retrieve_custom_trainer_state_dict,
)
from nemo_aligner.utils.utils import load_from_nemo
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer

from dataset import LLMJPSFTDataset, build_dataloader, load_datasets

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)


def get_latest_checkpoint(checkpoints_dir: str) -> Optional[str]:
    if not os.path.exists(checkpoints_dir):
        return None
    checkpoint_dirs: list[str] = [
        d for d in os.listdir(checkpoints_dir) if d.startswith("step=")
    ]
    if not checkpoint_dirs:
        return None
    latest_checkpoint = max(checkpoint_dirs, key=lambda d: int(d.split("=")[1]))
    return os.path.join(checkpoints_dir, latest_checkpoint)


def _modify_config(gpt_cfg, cfg, add_cfg_to_tree=False):
    """
    This function modifies the original gpt pre-training config (gpt_cfg) with attributes from the finetuning config (cfg).
    The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
    """
    OmegaConf.set_struct(gpt_cfg, True)
    OmegaConf.resolve(cfg)
    with open_dict(gpt_cfg):
        gpt_cfg.megatron_amp_O2 = cfg.model.get("megatron_amp_O2", False)
        gpt_cfg.micro_batch_size = cfg.mbs
        gpt_cfg.global_batch_size = cfg.gbs
        gpt_cfg.data = cfg.data
        gpt_cfg.sequence_parallel = cfg.model.get("sequence_parallel", False)
        gpt_cfg.activations_checkpoint_granularity = cfg.model.get(
            "activations_checkpoint_granularity", None
        )
        gpt_cfg.activations_checkpoint_num_layers = cfg.model.get(
            "activations_checkpoint_num_layers", None
        )
        gpt_cfg.activations_checkpoint_method = cfg.model.get(
            "activations_checkpoint_method", None
        )
        gpt_cfg.activations_checkpoint_layers_per_pipeline = cfg.model.get(
            "activations_checkpoint_layers_per_pipeline", None
        )
        gpt_cfg.peft = cfg.model.peft
        gpt_cfg.optim = cfg.model.optim
        gpt_cfg.precision = cfg.trainer.precision
        gpt_cfg.restore_from_path = cfg.model.restore_from_path
        gpt_cfg.resume_from_checkpoint = cfg.model.resume_from_checkpoint
        gpt_cfg.save_nemo_on_validation_end = cfg.model.save_nemo_on_validation_end
        gpt_cfg.gradient_as_bucket_view = cfg.model.gradient_as_bucket_view
        gpt_cfg.hidden_dropout = cfg.model.get("hidden_dropout", 0.0)
        gpt_cfg.attention_dropout = cfg.model.get("attention_dropout", 0.0)
        gpt_cfg.ffn_dropout = cfg.model.ffn_dropout
        gpt_cfg.use_flash_attention = cfg.model.get("use_flash_attention", False)
        # if TP/PP size is -1, use default TP/PP size as original model
        if cfg.model.get("tensor_model_parallel_size", 1) > 0:
            gpt_cfg.tensor_model_parallel_size = cfg.model.get(
                "tensor_model_parallel_size", 1
            )
        if cfg.model.get("pipeline_model_parallel_size", 1) > 0:
            gpt_cfg.pipeline_model_parallel_size = cfg.model.get(
                "pipeline_model_parallel_size", 1
            )
        gpt_cfg.pipeline_model_parallel_split_rank = cfg.model.get(
            "pipeline_model_parallel_split_rank", 0
        )
        gpt_cfg.use_loss_mask = cfg.model.use_loss_mask

        sft_cls = GPTSFTModel
        gpt_cfg.target = f"{sft_cls.__module__}.{sft_cls.__name__}"

        if cfg.model.get("use_flash_attention", None) is not None:
            gpt_cfg.use_flash_attention = cfg.model.use_flash_attention

        if cfg.model.get("seq_len_interpolation_factor", None) is not None:
            gpt_cfg.seq_len_interpolation_factor = (
                cfg.model.seq_len_interpolation_factor
            )

        # This is needed when modifying a hparam file directly to load `.ckpt` files.
        # This is not needed to modify the cfg in `.nemo` files.
        if add_cfg_to_tree:
            OmegaConf.resolve(gpt_cfg)
            gpt_cfg.cfg = gpt_cfg

    return gpt_cfg


@hydra_runner(config_path="configs", config_name="sft")
def main(cfg):
    if is_global_rank_zero():
        logging.info("\n\n************** Experiment configuration ***********")
        logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    if cfg.use_mpi:
        global_rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
        local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 1))
        os.environ["RANK"] = str(global_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        if cfg.use_slurm:
            os.environ["SLURM_PROCID"] = str(global_rank)
            os.environ["SLURM_LOCALID"] = str(local_rank)
            os.environ["SLURM_NTASKS"] = str(world_size)
        logging.info(
            f"global_rank: {global_rank}, local_rank: {local_rank}, world_size: {world_size}"
        )

    trainer: Trainer = resolve_and_create_trainer(cfg, "sft")
    log_dir = exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    ptl_model, updated_cfg = load_from_nemo(
        GPTSFTModel,
        model_cfg=cfg,
        trainer=trainer,
        strict=True,
        modify_config_fn=_modify_config,
        restore_path=cfg.model.restore_from_path,
        return_updated_cfg=True,
    )
    init_peft(ptl_model, updated_cfg)

    latest_checkpoint: Optional[str] = get_latest_checkpoint(f"{log_dir}/checkpoints")
    if latest_checkpoint is not None:
        logging.info(f"Resuming from checkpoint: {latest_checkpoint}")
        custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)
        consumed_samples: int = custom_trainer_state_dict["consumed_samples"]
    else:
        logging.info("No checkpoint found. Starting from scratch.")
        custom_trainer_state_dict = None
        consumed_samples = 0

    # save the updated config to the log directory
    if is_global_rank_zero():
        updated_config_path: str = f"{log_dir}/checkpoints/model_config.yaml"
        os.makedirs(os.path.dirname(updated_config_path), exist_ok=True)
        OmegaConf.save(updated_cfg, updated_config_path)

    with open_dict(cfg):
        # overwrite the model config with the config from the checkpoint
        cfg.model.encoder_seq_length = ptl_model.cfg.encoder_seq_length

    train_examples, dev_examples = load_datasets(
        cfg, res_quality_threshold=cfg.res_quality_threshold
    )

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    train_dataset = LLMJPSFTDataset(
        loaded_examples=train_examples,
        tokenizer=ptl_model.tokenizer,
        use_loss_mask=cfg.model.use_loss_mask,
        max_seq_length=cfg.model.max_seq_length,
    )
    val_dataset = LLMJPSFTDataset(
        loaded_examples=dev_examples,
        tokenizer=ptl_model.tokenizer,
        use_loss_mask=cfg.model.use_loss_mask,
        max_seq_length=cfg.model.max_seq_length,
    )

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        consumed_samples=consumed_samples,
        micro_batch_size=cfg.data.train_ds.micro_batch_size,
        global_batch_size=cfg.data.train_ds.global_batch_size,
        collate_fn=train_dataset.collate_fn,
        seed=cfg.seed,
    )
    val_dataloader = build_dataloader(
        dataset=val_dataset,
        consumed_samples=0,
        micro_batch_size=cfg.data.validation_ds.micro_batch_size,
        global_batch_size=cfg.data.validation_ds.global_batch_size,
        collate_fn=val_dataset.collate_fn,
    )

    init_using_ptl(trainer, ptl_model, train_dataloader, train_dataset)
    optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(ptl_model)

    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)

    logger.log_hyperparams(OmegaConf.to_container(cfg))
    timer = Timer(cfg.exp_manager.get("max_time_per_run"))

    sft_trainer = SupervisedTrainer(
        cfg=cfg.trainer.sft,
        model=ptl_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=None,
        logger=logger,
        ckpt_callback=ckpt_callback,
        run_timer=timer,
    )

    if custom_trainer_state_dict is not None:
        sft_trainer.load_state_dict(custom_trainer_state_dict)

    sft_trainer.fit()

    # remove optimizer state files
    for optimizer_state_file in glob.glob(
        f"{log_dir}/checkpoints/step*/optimizer.state.*"
    ):
        try:
            shutil.rmtree(optimizer_state_file)
            logging.info(f"Deleted directory: {optimizer_state_file}")
        except OSError as e:
            logging.error(f"Error: {optimizer_state_file} : {e.strerror}")


if __name__ == "__main__":
    main()
