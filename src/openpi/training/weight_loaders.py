import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")




@dataclasses.dataclass(frozen=True)
class ValueModelWeightLoader(WeightLoader):
    """加载 SigLIP 和 Gemma 3 270M 预训练权重用于 ValueModel。

    - SigLIP: 
    - Gemma 3 270M: 
    - ValueHead: 随机初始化
    """

    gemma_variant: str = "gemma3-270m"

    def load(self, params: at.Params) -> at.Params:
        logger.info("加载 SigLIP 权重 (from local checkpoint)...")
        #siglip_path = "/data/train_dataset/checkpoint/siglip2-so400m-patch14-384-jax/siglip2_so400m14_384.npz"
        siglip_path = "/public/home/wangsenbao/Robotic_Project/checkpoint/siglip2-so400m-patch14-384-jax/siglip2_so400m14_384.npz"
        with open(siglip_path, "rb") as f:
            siglip_flat = dict(np.load(f, allow_pickle=False))
        siglip_params = flax.traverse_util.unflatten_dict(siglip_flat, sep="/")["params"]

        logger.info("加载 Gemma 3 270M 权重 (from local Orbax checkpoint)...")
        gemma_checkpoint_dir = "/public/home/wangsenbao/Robotic_Project/checkpoint/gemma-3-270m"
        #gemma_checkpoint_dir = "/data/train_dataset/checkpoint/gemma-3-270m"

        # 使用 Orbax 加载 checkpoint
        try:
            from orbax.checkpoint import PyTreeCheckpointer

            checkpointer = PyTreeCheckpointer()
            gemma_params = checkpointer.restore(gemma_checkpoint_dir)
            logger.info("成功加载 Gemma checkpoint")
        except Exception as e:
            logger.error(f"Gemma 加载失败: {e}")
            raise

        loaded_params = {
            "img": siglip_params,
            "llm": gemma_params,
        }

        return _merge_params(loaded_params, params, missing_regex=".*value_head.*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
