# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import ast
from hmac import trans_36
import os
import copy
from dataclasses import dataclass, field
import json
import logging    
import pathlib
from typing import Dict, Optional, Sequence, List
from PIL import Image, ImageFile
from packaging import version
import numpy as np
from functools import partial

import time
import random
import yaml
import math
import re
import torch
import glob
from torchvision.transforms import v2

import transformers
import tokenizers

from transformers import AutoConfig
from torch.utils.data import Dataset
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token
from llava.utils import rank0_print, process_video_with_pyav, process_video_with_decord

from streamvln.model.stream_video_vln import StreamVLNForCausalLM
from streamvln.dataset.vln_action_dataset import collate_fn, VLNActionDataset

from streamvln.utils.utils import ANSWER_LIST, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX, DEFAULT_VIDEO_TOKEN
torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")

from streamvln.args import ModelArguments, DataArguments, TrainingArguments

# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
# from peft.tuners.lora import LoraLayer
import torch.nn as nn
import torch.nn.functional as F
from streamvln.model.continual_learning import ContinualLearningTrainer, RouteManager, EWCLoss
from streamvln.model.tucker_lora_layers import Tucker4DLoRALinear, Tucker4DLoRALayer

from streamvln.model.triple_distillation import TripleDistillationLoss
from streamvln.model.continual_learning import ProtoStreamCL  # 新增的ProtoStreamCL类

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

        
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["vision_tower",
                            "mm_projector",
                            "mem_projector",
                            "point_projector",
                            "vision_resampler",
                            "mem_resampler",
                            "pointnet"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, model_args=None):
    """Collects the state dict and dump to disk."""
    
    if hasattr(trainer.args, 'use_lora') and trainer.args.use_lora:
        rank0_print("Saving Tucker-LoRA weights only...")
        
        tucker_dir = os.path.join(output_dir, "tucker_lora")
        os.makedirs(tucker_dir, exist_ok=True)
        
        # Collect ALL Tucker-LoRA parameters, regardless of requires_grad
        tucker_state_dict = {}
        
        for name, param in trainer.model.named_parameters():
            if 'lora_layer' in name:  # Remove the requires_grad check
                # Save ALL Tucker layer parameters
                tucker_state_dict[name] = param.data.detach().cpu()
        
        if trainer.args.should_save:
            # Save Tucker weights
            torch.save(tucker_state_dict, os.path.join(tucker_dir, "tucker_lora_weights.bin"))
            
            # Save config
            trainer.model.config.save_pretrained(output_dir)
            
            # Save metadata
            metadata = {
                "model_type": "streamvln_tucker_lora",
                "base_model_path": model_args.model_name_or_path if model_args else trainer.args.model_name_or_path,
                "task_id": trainer.args.current_task_id,
                "tucker_lora_config": {
                    "scene_num": trainer.args.tucker_scene_num,
                    "env_num": trainer.args.tucker_env_num,
                    "ranks": trainer.args.tucker_ranks_4d,
                    "alpha": trainer.args.lora_alpha,
                },
            }
            
            with open(os.path.join(output_dir, "tucker_lora_config.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            rank0_print(f"Tucker-LoRA weights saved to {tucker_dir}")
            rank0_print(f"Saved {len(tucker_state_dict)} Tucker parameters:")
            
            # Log which parameters were saved
            u1_params = [k for k in tucker_state_dict if '.U1' in k]
            u2_params = [k for k in tucker_state_dict if '.U2' in k]
            u3_params = [k for k in tucker_state_dict if '.U3' in k]
            u4_params = [k for k in tucker_state_dict if '.U4' in k]
            g_params = [k for k in tucker_state_dict if '.G' in k]
            
            rank0_print(f"  - G parameters: {len(g_params)}")
            rank0_print(f"  - U1 parameters: {len(u1_params)}")
            rank0_print(f"  - U2 parameters: {len(u2_params)}")
            rank0_print(f"  - U3 parameters: {len(u3_params)}")
            rank0_print(f"  - U4 parameters: {len(u4_params)}")
        
        return

    if hasattr(trainer.args, "tune_mm_mlp_adapter") and trainer.args.tune_mm_mlp_adapter:
        check_only_save_mm_adapter_tunnable = True
    elif hasattr(trainer.args, "mm_tunable_parts") and (len(trainer.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in trainer.args.mm_tunable_parts or "mm_vision_resampler" in trainer.args.mm_tunable_parts)):
        check_only_save_mm_adapter_tunnable = True
    else:
        check_only_save_mm_adapter_tunnable = False

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")
                
    if check_only_save_mm_adapter_tunnable:
        # Only save Adapter
        keys_to_match = ["mm_projector", "vision_resampler"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        return

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def safe_save_model_for_hf_trainer_fsdp(trainer: transformers.Trainer,
                                   output_dir: str, model_args=None):
    """Save model with explicit LoRA separation for FSDP"""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    
    if hasattr(trainer.args, 'use_lora') and trainer.args.use_lora:
        rank0_print("Saving Tucker-LoRA weights only (FSDP)...")
        
        tucker_dir = os.path.join(output_dir, "tucker_lora")
        
        if trainer.args.should_save:
            os.makedirs(tucker_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
        
        # Get full state dict using FSDP
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            trainer.model, StateDictType.FULL_STATE_DICT, save_policy
        ):
            state_dict = trainer.model.state_dict()
        
        # Collect ALL Tucker-LoRA parameters (remove requires_grad check)
        tucker_state_dict = {}
        
        for name, param in state_dict.items():
            if 'lora_layer' in name:  # Save ALL lora_layer parameters
                tucker_state_dict[name] = param.detach().cpu()
        
        if trainer.args.should_save:
            # Save Tucker weights
            torch.save(tucker_state_dict, os.path.join(tucker_dir, "tucker_lora_weights.bin"))
            
            # Save model config
            trainer.model.config.save_pretrained(output_dir)
            
            # Save metadata with detailed info
            metadata = {
                "model_type": "streamvln_tucker_lora",
                "base_model_path": model_args.model_name_or_path if model_args else trainer.args.model_name_or_path,
                "task_id": trainer.args.current_task_id,
                "tucker_lora_config": {
                    "scene_num": trainer.args.tucker_scene_num,
                    "env_num": trainer.args.tucker_env_num,
                    "ranks": trainer.args.tucker_ranks_4d,
                    "alpha": trainer.args.lora_alpha,
                    "target_modules": trainer.args.lora_target_modules,
                    "dropout": trainer.args.lora_dropout,
                    "init_scale": getattr(trainer.args, 'tucker_init_scale', 0.02),
                },
                "training_type": "fsdp"
            }
            
            with open(os.path.join(output_dir, "tucker_lora_config.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Log saved parameters
            rank0_print(f"Tucker-LoRA weights saved to {tucker_dir}")
            rank0_print(f"Saved {len(tucker_state_dict)} Tucker parameters:")
            
            u1_params = [k for k in tucker_state_dict if '.U1' in k]
            u2_params = [k for k in tucker_state_dict if '.U2' in k]
            u3_params = [k for k in tucker_state_dict if '.U3' in k]
            u4_params = [k for k in tucker_state_dict if '.U4' in k]
            g_params = [k for k in tucker_state_dict if '.G' in k]
            
            rank0_print(f"  - G parameters: {len(g_params)}")
            rank0_print(f"  - U1 parameters: {len(u1_params)}")
            rank0_print(f"  - U2 parameters: {len(u2_params)}")
            rank0_print(f"  - U3 parameters: {len(u3_params)}")
            rank0_print(f"  - U4 parameters: {len(u4_params)}")
        
        return

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.state_dict_type = "FULL_STATE_DICT"
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # TODO maybe this should be changed for interleaved data?
            # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
            # only check for num_im=1
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources


def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(sources: List[List[Dict[str, str]]], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv: conversation_lib.Conversation = conversation_lib.default_conversation.copy()
    roles: Dict[str, str] = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations: List[str] = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source: List[Dict[str, str]] = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role: str = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids: torch.Tensor = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids: torch.Tensor = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets: torch.Tensor = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask target
    sep: str = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: List[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))

        cur_len = 1  # Ignore <bos>
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Re-append sep because split on this
            # Now "".join(parts)==rou

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  # Ignore <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # Ignore <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # Ignore <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # Ignore <bos>

            round_len += 2  # sep: <end_of_turn>\n takes 2 tokens
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
            
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    # import ipdb; ipdb.set_trace()
    # tmp = []
    # for id in input_ids[0].cpu().numpy():
    #     try:
    #         print(tokenizer.convert_ids_to_tokens([id]))
    #         tmp.extend(tokenizer.convert_ids_to_tokens([id]))
    #     except:
    #         print(id) 
    # print(' '.join(tmp))
    # # exit()
    # for id in targets[0].cpu().numpy():
    #     try:
    #         print(tokenizer.convert_ids_to_tokens([id]))
    #     except:
    #         print(id)
    # exit()

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f"(#turns={len(re_rounds)} ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class CombineDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        super(CombineDataset, self).__init__()
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cum_lengths = np.cumsum(self.lengths)

    def __len__(self):
        return self.cum_lengths[-1]
    
    @property
    def task_lengths(self):
        ##nav task id : 0
        ##scanqa task id: 1
        ##vqa task id: 2
        ##vcap task id: 3
        res = []
        for dataset in self.datasets:
            if hasattr(dataset, "task"):
                task = dataset.task
            else:
                task = 0
            res.extend([(task, 1) for i in range(len(dataset))])
        return res

    def __getitem__(self, i):
        for idx, cum_len in enumerate(self.cum_lengths):
            if i < cum_len:
                return self.datasets[idx][i - cum_len + self.lengths[idx]]
        raise ValueError(f"Index {i} out of bound")

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.list_data_dict = []

        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list

    def process_image(self, image_file, overwrite_image_aspect_ratio=None):
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
                # Handling multi images
                # overwrite to process with simple pad 
                if len(image_file) > 1:
                    image = [self.process_image(f, "pad") for f in image_file]
                    image = [[im[0], im[1], "image"] for im in image]
            else:
                image = [self.process_image(image_file)]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        elif "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.data_args.video_folder
            video_file = os.path.join(video_folder, video_file)
            suffix = video_file.split(".")[-1]
            if not os.path.exists(video_file):
                print("File {} not exist!".format(video_file))

            try:
                if "shareVideoGPTV" in video_file:
                    frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
                    frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

                    # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
                    if self.data_args.force_sample:
                        num_frames_to_sample = self.data_args.frames_upbound
                    else:
                        num_frames_to_sample = 10

                    avg_fps = 2
                    
                    total_frames = len(frame_files)
                    sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)


                    frame_time = [i/2 for i in sampled_indices]
                    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

                    video_time = total_frames / avg_fps

                    # Read and store the sampled frames
                    video = []
                    for idx in sampled_indices:
                        frame_path = frame_files[idx]
                        try:
                            with Image.open(frame_path) as img:
                                frame = img.convert("RGB")
                                video.append(frame)
                        except IOError:
                            print(f"Failed to read frame at path: {frame_path}")
                else:
                    video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, self.data_args)

                processor = self.data_args.image_processor
                image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                if self.data_args.add_time_instruction:
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                    sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
                image = [(image, video[0].size, "video")]
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                # print(sources)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Failed to read video file: {video_file}")
                return self._get_item(i + 1)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = ("image" in self.list_data_dict[i]) or ("video" in self.list_data_dict[i])
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif "video" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text"),
            ]
        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = self.list_data_dict[i].get("id", i)

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def pad_tensors(tensors, lens=None, max_len=None, pad=0):
        """B x [T, ...]"""

        if lens is None:
            lens = [t.size(0) for t in tensors]
            if len(lens) == 1 and lens[0] == max_len:
                return tensors
        if max_len is None:
            max_len = max(lens)
        bs = len(tensors)
        hid = tensors[0].shape[1:]
        dtype = tensors[0].dtype
        output = torch.zeros(bs, max_len, *hid, dtype=dtype).to(tensors[0].device)
        if pad:
            output.data.fill_(pad)
        for i, (t, l) in enumerate(zip(tensors, lens)):
            output.data[i, :l, ...] = t.data
        return output

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        (input_ids_batch, labels_batch, image_batch, depth_batch, 
        pose_batch, intrinsic_batch, time_ids_batch, inflection_weight_batch) = \
            tuple([instance.get(key, None) for instance in instances] 
                for key in ("input_ids", "labels", "images", "depths", 
                            "poses", "intrinsics", "time_ids", "inflection_weights"))
        
        # Filtering None values
        input_ids_batch = [x for x in input_ids_batch if x is not None]
        labels_batch = [x for x in labels_batch if x is not None]
        
        # Pad input_ids and labels
        input_ids_batch = self.pad_sequence(input_ids_batch, batch_first=True, 
                                            padding_value=self.tokenizer.pad_token_id)
        labels_batch = self.pad_sequence(labels_batch, batch_first=True, 
                                        padding_value=IGNORE_INDEX)
        input_ids_batch = input_ids_batch[:, :self.tokenizer.model_max_length]
        labels_batch = labels_batch[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids_batch.ne(self.tokenizer.pad_token_id)
        
        # Processing image and depth data
        if image_batch[0] is not None:
            # Ensure all image tensors have consistent shape
            img_lens = [img.shape[0] if img is not None else 0 for img in image_batch]
            max_len = max(img_lens) if img_lens else 1
            
            # Fill image and depth tensor
            padded_images = []
            padded_depths = []
            
            for i, (img, depth) in enumerate(zip(image_batch, depth_batch)):
                if img is not None and depth is not None:
                   
                    if not isinstance(img, torch.Tensor):
                        img = torch.tensor(img)
                    if not isinstance(depth, torch.Tensor):
                        depth = torch.tensor(depth)
                        
                    # Fill to maximum length
                    if img.shape[0] < max_len:
                        pad_size = max_len - img.shape[0]
                        img = torch.cat([img, torch.zeros(pad_size, *img.shape[1:], dtype=img.dtype, device=img.device)], dim=0)
                        depth = torch.cat([depth, torch.zeros(pad_size, *depth.shape[1:], dtype=depth.dtype, device=depth.device)], dim=0)
                        
                    padded_images.append(img)
                    padded_depths.append(depth)
                else:
                    # Creating a dummy tensor
                    if padded_images:
                        dummy_img = torch.zeros_like(padded_images[0])
                        dummy_depth = torch.zeros_like(padded_depths[0])
                    else:
                        dummy_img = torch.zeros(max_len, 3, 224, 224)
                        dummy_depth = torch.zeros(max_len, 1, 224, 224)
                    padded_images.append(dummy_img)
                    padded_depths.append(dummy_depth)
            
            image_batch = torch.stack(padded_images) if padded_images else None
            depth_batch = torch.stack(padded_depths) if padded_depths else None
        else:
            image_batch = None
            depth_batch = None
        
        # Handling poses and intrinsics
        if pose_batch[0] is not None:
            padded_poses = []
            padded_intrinsics = []
            max_views = max([p.shape[0] if p is not None else 0 for p in pose_batch])
            
            for pose, intrinsic in zip(pose_batch, intrinsic_batch):
                if pose is not None and intrinsic is not None:
                    view = pose.shape[0]
                    pad_view = max_views - view
                    if pad_view > 0:
                        padded_pose = torch.cat([pose, torch.eye(4).unsqueeze(0).repeat(pad_view, 1, 1)], dim=0)
                        padded_intrinsic = torch.cat([intrinsic, intrinsic[0:1].repeat(pad_view, 1, 1)], dim=0)
                    else:
                        padded_pose = pose
                        padded_intrinsic = intrinsic
                    padded_poses.append(padded_pose)
                    padded_intrinsics.append(padded_intrinsic)
                else:
                    # Creating a dummy tensor
                    dummy_pose = torch.eye(4).unsqueeze(0).repeat(max_views, 1, 1)
                    dummy_intrinsic = torch.eye(3).unsqueeze(0).repeat(max_views, 1, 1)
                    padded_poses.append(dummy_pose)
                    padded_intrinsics.append(dummy_intrinsic)
            
            pose_batch = torch.stack(padded_poses) if padded_poses else None
            intrinsic_batch = torch.stack(padded_intrinsics) if padded_intrinsics else None
        else:
            pose_batch = None
            intrinsic_batch = None
        
        # Build and return batch
        batch = {
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            "attention_mask": attention_mask,
        }
        
        if image_batch is not None:
            batch["images"] = image_batch
        if depth_batch is not None:
            batch["depths"] = depth_batch
        if pose_batch is not None:
            batch["poses"] = pose_batch
        if intrinsic_batch is not None:
            batch["intrinsics"] = intrinsic_batch
        
        # Processing time_ids and weights
        if time_ids_batch[0] is not None:
            valid_time_ids = [x for x in time_ids_batch if x is not None]
            if valid_time_ids:
                batch["time_ids"] = valid_time_ids
        
        if inflection_weight_batch[0] is not None:
            valid_weights = [x for x in inflection_weight_batch if x is not None]
            if valid_weights:
                batch["inflection_weights"] = torch.stack(valid_weights)
        
        return batch
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # # input_ids, labels, image_batch, depth_batch, pose_batch, intrinsic_batch, time_ids_batch, inflection_weight_batch = zip(
        # #     *[(inst["input_ids"], inst["labels"], inst["image"], inst["depth"], inst["pose"], inst["intrinsic"], inst["time_ids"], inst["inflection_weight"])
        # #       for inst in instances]
        # # )
        # # input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "id"))
        # input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        # labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        # if self.tokenizer.pad_token_id is None:
        #     # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
        #     self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        # input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        # batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        # # batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), ids=ids)

        # if "image" in instances[0]:
        #     images = [instance["image"] for instance in instances]

        #     batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
        #     batch["modalities"] = [im[2] for im_list in images for im in im_list]
        #     images = [im[0] for im_list in images for im in im_list]

        #     # if all(x is not None and x.shape == images[0].shape for x in images):
        #         # Image: (N, P, C, H, W)
        #         # Video: (N, F, C, H, W)
        #     #     batch["images"] = torch.stack(images)
        #     # else:
        #     batch["images"] = images

        # if "prompt" in instances[0]:
        #     batch["prompts"] = [instance["prompt"] for instance in instances]

        # return batch
        
        
def load_tucker_weights(model, tucker_state_dict):
    """Load Tucker weights to model"""
    missing_keys = []
    unexpected_keys = []
    loaded_keys = []
    
    model_state_dict = model.state_dict()
    
    for name, param in tucker_state_dict.items():
        if name in model_state_dict:
            # Check shape match
            if model_state_dict[name].shape != param.shape:
                rank0_print(f"Warning: Shape mismatch for {name}: "
                          f"model {model_state_dict[name].shape} vs checkpoint {param.shape}")
                unexpected_keys.append(name)
                continue
            
            # Load weights
            model_state_dict[name].copy_(param.to(model_state_dict[name].device).to(model_state_dict[name].dtype))
            loaded_keys.append(name)
        else:
            unexpected_keys.append(name)
    
    # Check which Tucker parameters in model weren't loaded
    for name, param in model.named_parameters():
        if 'lora_layer' in name and name not in loaded_keys:
            missing_keys.append(name)
    
    rank0_print(f"Loaded {len(loaded_keys)} Tucker parameters")
    
    # Count by parameter type
    u1_loaded = len([k for k in loaded_keys if '.U1' in k])
    u2_loaded = len([k for k in loaded_keys if '.U2' in k])
    u3_loaded = len([k for k in loaded_keys if '.U3' in k])
    u4_loaded = len([k for k in loaded_keys if '.U4' in k])
    g_loaded = len([k for k in loaded_keys if '.G' in k])
    
    rank0_print(f"  - G: {g_loaded} loaded")
    rank0_print(f"  - U1: {u1_loaded} loaded")
    rank0_print(f"  - U2: {u2_loaded} loaded")
    rank0_print(f"  - U3: {u3_loaded} loaded")
    rank0_print(f"  - U4: {u4_loaded} loaded")
    
    if missing_keys:
        rank0_print(f"Missing {len(missing_keys)} parameters (will be initialized)")
        # Only show first few missing keys
        for key in missing_keys[:5]:
            rank0_print(f"  - {key}")
    if unexpected_keys:
        rank0_print(f"Unexpected {len(unexpected_keys)} keys in checkpoint")
    
    return {'missing': missing_keys, 'unexpected': unexpected_keys, 'loaded': loaded_keys}

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,vision_tower, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    nav_dataset = VLNActionDataset(tokenizer=tokenizer, data_args=data_args, task_id=0)
    dataset =[nav_dataset]
        
    if len(dataset) > 1:
        train_dataset = CombineDataset(dataset)
    else:
        train_dataset = dataset[0]
        
    print('len train_dataset ', len(train_dataset))

    data_collator = partial(collate_fn, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def get_model(model_args, training_args, data_args, bnb_model_from_pretrained_args):
    # import ipdb; ipdb.set_trace()
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    cfg_pretrained = None

    overwrite_config = {}
    if any(
        [
            model_args.rope_scaling_factor is not None,
            model_args.rope_scaling_type is not None,
            model_args.mm_spatial_pool_stride is not None,
            model_args.mm_spatial_pool_out_channels is not None,
            model_args.mm_spatial_pool_mode is not None,
            model_args.mm_resampler_type is not None,
        ]
    ):
        cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    # import ipdb; ipdb.set_trace()
    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )

    if model_args.mm_spatial_pool_stride is not None and model_args.mm_spatial_pool_out_channels is not None and model_args.mm_spatial_pool_mode is not None and model_args.mm_resampler_type is not None:
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = model_args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.mm_spatial_pool_mode is not None:
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode
    if model_args.mm_spatial_pool_size is not None:
        overwrite_config["mm_spatial_pool_size"] = model_args.mm_spatial_pool_size

    if data_args.num_future_steps:
        overwrite_config["num_future_steps"] = data_args.num_future_steps
    if data_args.num_history:
        overwrite_config["num_history"] = data_args.num_history
        
    if model_args.mm_tunable_parts:
        overwrite_config["mm_tunable_parts"] = model_args.mm_tunable_parts
    
    overwrite_config["mm_newline_position"] = model_args.mm_newline_position
    overwrite_config["mm_patch_merge_type"] = model_args.mm_patch_merge_type
   
    if overwrite_config:
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)

        customized_kwargs["config"] = cfg_pretrained

    if hasattr(training_args, 'use_protostream') and training_args.use_protostream:
        rank0_print("=" * 50)
        rank0_print("[ProtoStream] Configuring ProtoStream parameters...")
        rank0_print("=" * 50)
        
        # Ensure that a config object exists.
        if 'config' not in customized_kwargs or customized_kwargs['config'] is None:
            customized_kwargs['config'] = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                trust_remote_code=True
            )
        
        config = customized_kwargs['config']
        
        config.use_protostream = True
        config.proto_dim = getattr(training_args, 'proto_dim', 512)
        config.similarity_threshold = getattr(training_args, 'similarity_threshold', 0.7)
        config.max_prototypes = getattr(training_args, 'max_prototypes', 100)
        config.proto_momentum = getattr(training_args, 'proto_momentum', 0.9)
        config.diversity_weight = getattr(training_args, 'diversity_weight', 0.1)
        config.temperature = getattr(training_args, 'temperature', 1.0)
        
        # HyperLoRA parameters
        config.lora_rank = getattr(training_args, 'hypernet_lora_rank', 16)
        config.lora_alpha = getattr(training_args, 'hypernet_lora_alpha', 32.0)
        config.lora_dropout = getattr(training_args, 'hypernet_lora_dropout', 0.1)
        
        # Target modules
        lora_target_modules = getattr(training_args, 'lora_target_modules', 'q_proj,v_proj')
        if isinstance(lora_target_modules, str):
            config.lora_target_modules = lora_target_modules.split(',')
        else:
            config.lora_target_modules = lora_target_modules
        
        rank0_print(f"  - use_protostream: {config.use_protostream}")
        rank0_print(f"  - proto_dim: {config.proto_dim}")
        rank0_print(f"  - max_prototypes: {config.max_prototypes}")
        rank0_print(f"  - lora_rank: {config.lora_rank}")
        rank0_print(f"  - lora_target_modules: {config.lora_target_modules}")
        rank0_print("[ProtoStream] Configuration complete ✓")
        rank0_print("=" * 50)


    model = StreamVLNForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
                )
    
    # # LoRA
    # if training_args.use_lora:
    #     rank0_print("Setting up explicit LoRA layers...")
    #     model = apply_lora_to_model(model, training_args)

    if training_args.use_lora:
        rank0_print("Setting up explicit LoRA layers...")
        model = apply_lora_to_model(model, training_args)

# Add a new training loop wrapper function
def train_with_continual_learning(trainer, training_args, data_module):
    """Continual learning training loop"""
    from streamvln.model.continual_learning import ContinualLearningTrainer
    from streamvln.model.tucker_lora_layers import Tucker4DLoRALayer, Tucker4DLoRALinear
    import os
    import glob

    # Check model data type
    model_dtype = next(trainer.model.parameters()).dtype
    print(f"Model dtype: {model_dtype}")
    
    # Initialize continual learning trainer
    cl_trainer = ContinualLearningTrainer(
        trainer.model, 
        training_args, 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Set route for current task
    scene_idx, env_idx = cl_trainer.set_task_route(training_args.current_task_id)
    
    # Set route for all Tucker4DLoRALinear layers
    for name, module in trainer.model.named_modules():
        if isinstance(module, Tucker4DLoRALinear):
            module.set_route(scene_idx, env_idx, training_args.current_task_id)
    
    # Load Fisher matrix from previous tasks if exists
    if training_args.current_task_id > 0 and training_args.fisher_path:
        for prev_task_id in range(training_args.current_task_id - 1, -1, -1):
            prev_fisher_path = os.path.join(
                training_args.fisher_path,
                f"fisher_task_{prev_task_id}.pt"
            )
            if os.path.exists(prev_fisher_path):
                cl_trainer.ewc_loss.load_fisher_matrix(prev_fisher_path)
                rank0_print(f"Loaded Fisher matrix from task {prev_task_id}")
                break
    
    # Custom callback for gradient masking
    class GradientMaskingCallback(transformers.TrainerCallback):
        def __init__(self, cl_trainer, trainer):
            self.cl_trainer = cl_trainer
            self.trainer = trainer
            self.step_count = 0
            
        def on_before_optimizer_step(self, args, state, control, **kwargs):
            """Apply gradient masking before optimizer step"""
            for name, module in self.trainer.model.named_modules():
                if isinstance(module, Tucker4DLoRALayer):
                    module.zero_inactive_gradients()
            return control
        
        def on_step_end(self, args, state, control, **kwargs):
            """Logging after each training step"""
            if state.global_step % args.logging_steps == 0:
                loss_value = "N/A"
                if state.log_history and len(state.log_history) > 0:
                    for log_entry in reversed(state.log_history):
                        if isinstance(log_entry, dict) and 'loss' in log_entry:
                            loss_value = f"{log_entry['loss']:.4f}"
                            break
                
                print(f"Task {args.current_task_id} - "
                      f"Step {state.global_step} - "
                      f"Loss: {loss_value}")
            return control
        
        def on_epoch_end(self, args, state, control, **kwargs):
            """Processing after each epoch"""
            print(f"Task {args.current_task_id} - "
                  f"Epoch {state.epoch} completed")
            return control
    
    # Add callback
    callback = GradientMaskingCallback(cl_trainer, trainer)
    trainer.add_callback(callback)
    
    # Save original compute_loss method
    original_compute_loss = trainer.compute_loss
    
    def compute_loss_with_ewc(model, inputs, return_outputs=False):
        """Compute loss with EWC and orthogonal constraints"""
        # Compute original task loss
        if return_outputs:
            result = original_compute_loss(model, inputs, return_outputs=True)
            if isinstance(result, tuple):
                loss, outputs = result[0], result[1:] if len(result) > 1 else None
            else:
                loss = result.loss if hasattr(result, 'loss') else result
                outputs = result
        else:
            loss = original_compute_loss(model, inputs, return_outputs=False)
            outputs = None
        
        # Add EWC and orthogonal constraints
        total_loss = cl_trainer.compute_total_loss(loss)
        
        if return_outputs:
            if outputs is not None:
                if hasattr(outputs, 'loss'):
                    outputs.loss = total_loss
                return (total_loss,) + (outputs,) if isinstance(result, tuple) else outputs
            else:
                return (total_loss,)
        return total_loss
    
    # Replace compute_loss method
    trainer.compute_loss = compute_loss_with_ewc
    
    # Start training
    rank0_print(f"Starting continual learning for Task {training_args.current_task_id}")
    rank0_print(f"Route: Scene {scene_idx}, Environment {env_idx}")
    rank0_print(f"EWC Lambda: {training_args.ewc_lambda}")
    rank0_print(f"Orthogonal Reg Weight: {training_args.ortho_reg_weight}")
    
    # Execute training
    try:
        trainer.train()
    except Exception as e:
        rank0_print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    # Save current task's U3 and U4 parameters
    cl_trainer.ewc_loss.save_task_specific_params(scene_idx, env_idx)
    rank0_print(f"Saved U3[{scene_idx}] and U4[{env_idx}] parameters for future stability constraint")

    # CRITICAL: Ensure ALL Tucker parameters are set correctly before saving
    # This ensures U3 and U4 are saved even if they weren't actively trained
    for name, param in trainer.model.named_parameters():
        if 'lora_layer' in name:
            # ALL lora_layer parameters should be accessible for saving
            # We don't change requires_grad here, just ensure they exist
            pass

    # Compute Fisher Information Matrix for shared parameters (G, U1, U2)
    if cl_trainer.ewc_loss is not None and data_module['train_dataset'] is not None:
        rank0_print("Computing Fisher Information Matrix for shared parameters (G, U1, U2)...")
        
        # Create a small dataloader for Fisher matrix computation
        from torch.utils.data import DataLoader, Subset
        
        dataset = data_module['train_dataset']
        num_samples = min(20, len(dataset))
        indices = torch.randperm(len(dataset))[:num_samples].tolist()
        subset_dataset = Subset(dataset, indices)
        
        fisher_dataloader = DataLoader(
            subset_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=data_module['data_collator'],
            drop_last=False,
            pin_memory=False
        )
        
        # Clear cache before computing Fisher
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        try:
            # Ensure all Tucker layers have correct route
            for name, module in trainer.model.named_modules():
                if isinstance(module, Tucker4DLoRALinear):
                    module.set_route(scene_idx, env_idx, training_args.current_task_id)
            
            # Compute Fisher matrix (only for G, U1, U2)
            cl_trainer.ewc_loss.compute_fisher_matrix(fisher_dataloader, num_samples=20)
            
            # Validate Fisher matrix
            if len(cl_trainer.ewc_loss.fisher_dict) > 0:
                rank0_print(f"Successfully computed Fisher matrix for "
                        f"{len(cl_trainer.ewc_loss.fisher_dict)} shared parameters")
                
                # Save Fisher matrix
                if training_args.fisher_path:
                    fisher_save_path = os.path.join(
                        training_args.fisher_path,
                        f"fisher_task_{training_args.current_task_id}.pt"
                    )
                    try:
                        os.makedirs(os.path.dirname(fisher_save_path), exist_ok=True)
                        cl_trainer.ewc_loss.save_fisher_matrix(fisher_save_path)
                        rank0_print(f"Fisher matrix saved to {fisher_save_path}")
                    except Exception as save_error:
                        rank0_print(f"Failed to save Fisher matrix: {save_error}")
            else:
                rank0_print("Warning: Fisher matrix is empty!")
                # Create default Fisher matrix
                rank0_print("Creating default Fisher matrix for shared parameters...")
                for name, param in trainer.model.named_parameters():
                    if param.requires_grad:
                        clean_name = name.replace('module.', '') if 'module.' in name else name
                        if ('.lora_layer.G' in clean_name or 
                            '.lora_layer.U1' in clean_name or 
                            '.lora_layer.U2' in clean_name) and \
                        '.lora_layer.U3' not in clean_name and \
                        '.lora_layer.U4' not in clean_name:
                            cl_trainer.ewc_loss.fisher_dict[name] = torch.ones_like(
                                param, device=param.device
                            ) * 1e-4
                            cl_trainer.ewc_loss.optimal_params_dict[name] = param.data.clone()
        
        except Exception as e:
            rank0_print(f"Error in Fisher matrix computation: {e}")
            import traceback
            traceback.print_exc()
            rank0_print("Creating minimal Fisher matrix for shared parameters...")
            
            for name, param in trainer.model.named_parameters():
                if param.requires_grad:
                    clean_name = name.replace('module.', '') if 'module.' in name else name
                    if ('.lora_layer.G' in clean_name or 
                        '.lora_layer.U1' in clean_name or 
                        '.lora_layer.U2' in clean_name) and \
                    '.lora_layer.U3' not in clean_name and \
                    '.lora_layer.U4' not in clean_name:
                        cl_trainer.ewc_loss.fisher_dict[name] = torch.ones_like(
                            param, device=param.device
                        ) * 1e-5
                        cl_trainer.ewc_loss.optimal_params_dict[name] = param.data.clone()
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Save final checkpoint
    # final_checkpoint_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    # trainer.save_model(final_checkpoint_dir)
    # rank0_print(f"Saved final checkpoint to {final_checkpoint_dir}")
    
    # Save route configuration
    if training_args.route_config_file:
        cl_trainer.route_manager.save_config(training_args.route_config_file)
    
    rank0_print(f"Task {training_args.current_task_id} training completed")
    
    return trainer

def train_protostream(trainer, training_args, data_module, cl_manager, current_task_id, current_task_name):
    """
        ProtoStream Continuous Learning Training Function - Trains only the current task
            Args:
            trainer: HuggingFace Trainer object
            training_args: Training parameters
            data_module: Data module
            cl_manager: ProtoStreamCL manager
            current_task_id: Current task ID
            current_task_name: Current task name
    """
    import torch.distributed as dist
    import transformers
    import torch
    import os
    import json
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Save the original compute_loss
    original_compute_loss = trainer.compute_loss
    
    def compute_loss_with_protostream(model, inputs, return_outputs=False):

        # Calculate the original task loss
        if return_outputs:
            result = original_compute_loss(model, inputs, return_outputs=True)
            if isinstance(result, tuple):
                loss = result[0]
                outputs = result[1] if len(result) > 1 else None
            else:
                loss = result.loss if hasattr(result, 'loss') else result
                outputs = result
        else:
            loss = original_compute_loss(model, inputs, return_outputs=False)
            outputs = None
        
        # Add ProtoStream distillation loss
        if cl_manager is not None and outputs is not None:
            total_loss, loss_dict = cl_manager.compute_task_loss(inputs, outputs, task_loss=loss)
            
            # Update prototype
            if model.training:
                cl_manager.update_prototypes(inputs)
        else:
            total_loss = loss
        
        if return_outputs:
            if outputs is not None:
                if hasattr(outputs, 'loss'):
                    outputs.loss = total_loss
                return (total_loss, outputs) if isinstance(result, tuple) else outputs
            return (total_loss,)
        return total_loss
    
    # callback function
    class ProtoStreamCallback(transformers.TrainerCallback):
        def __init__(self, cl_manager, trainer, training_args, task_id):
            self.cl_manager = cl_manager
            self.trainer = trainer
            self.training_args = training_args
            self.task_id = task_id
            
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % args.logging_steps == 0:
                if hasattr(self.training_args, 'log_prototype_stats') and self.training_args.log_prototype_stats:
                    if hasattr(self.trainer.model, 'prototype_lib'):
                        proto_stats = self.trainer.model.prototype_lib.get_statistics()
                        if proto_stats and 'num_prototypes' in proto_stats:
                            rank0_print(f"[Step {state.global_step}] Prototypes: {proto_stats['num_prototypes']}")
            return control
        
        def on_epoch_end(self, args, state, control, **kwargs):
            rank0_print(f"Task {self.task_id} - Epoch {int(state.epoch)} completed")
            return control
    
    # Add callback
    callback = ProtoStreamCallback(cl_manager, trainer, training_args, current_task_id)
    trainer.add_callback(callback)
    
    # Replace compute_loss
    trainer.compute_loss = compute_loss_with_protostream

    if current_task_id > 0 and hasattr(training_args, 'pretrained_checkpoint_path') and training_args.pretrained_checkpoint_path:
        checkpoint_dir = training_args.pretrained_checkpoint_path
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"[Checkpoint] Loading from previous task: {checkpoint_dir}")
            print(f"{'='*60}")
        
        scene_path = os.path.join(checkpoint_dir, 'scene_encoder.pt')
        if os.path.exists(scene_path):
            try:
                scene_state = torch.load(scene_path, map_location='cpu')
                
                loaded_count = 0
                for name, param in trainer.model.scene_encoder.named_parameters():
                    if name in scene_state:
                        param.data = scene_state[name].to(param.device)
                        loaded_count += 1
                
                if rank == 0:
                    print(f" Loaded scene_encoder ({loaded_count} params)")
            except Exception as e:
                if rank == 0:
                    print(f" Failed to load scene_encoder: {e}")
        else:
            if rank == 0:
                print(f" scene_encoder.pt not found in {checkpoint_dir}")
        
        hypernet_path = os.path.join(checkpoint_dir, 'global_hypernet.pt')
        if os.path.exists(hypernet_path):
            try:
                hypernet_state = torch.load(hypernet_path, map_location='cpu')
                trainer.model.model.global_hypernet.load_state_dict(hypernet_state, strict=False)
                
                if rank == 0:
                    print(f" Loaded global_hypernet")
            except Exception as e:
                if rank == 0:
                    print(f"Failed to load global_hypernet: {e}")
        else:
            if rank == 0:
                print(f"global_hypernet.pt not found in {checkpoint_dir}")
        
        proto_path = os.path.join(checkpoint_dir, 'prototype_lib.pt')
        if os.path.exists(proto_path):
            try:
                proto_data = torch.load(proto_path, map_location='cpu')

                if 'prototypes' in proto_data:
                    trainer.model.prototype_lib.prototypes.data = proto_data['prototypes'].to(
                        trainer.model.prototype_lib.prototypes.device
                    )
                    num_protos = proto_data['prototypes'].shape[0]
                    
                    if rank == 0:
                        print(f" Loaded prototype_lib ({num_protos} prototypes)")
                
                if 'task_ids' in proto_data and proto_data['task_ids'] is not None:
                    if hasattr(trainer.model.prototype_lib, 'task_ids'):
                        trainer.model.prototype_lib.task_ids = proto_data['task_ids']
                
                if 'frozen_mask' in proto_data and proto_data['frozen_mask'] is not None:
                    if hasattr(trainer.model.prototype_lib, 'frozen_mask'):
                        trainer.model.prototype_lib.frozen_mask = proto_data['frozen_mask']
                        
            except Exception as e:
                if rank == 0:
                    print(f"  Failed to load prototype_lib: {e}")
        else:
            if rank == 0:
                print(f"  prototype_lib.pt not found in {checkpoint_dir}")
        
        cl_state_path = os.path.join(checkpoint_dir, 'cl_state.pt')
        if os.path.exists(cl_state_path):
            try:
                cl_state = torch.load(cl_state_path, map_location='cpu')
                
                if 'task_history' in cl_state and hasattr(cl_manager, 'task_history'):
                    cl_manager.task_history = cl_state['task_history']
                
                if rank == 0:
                    print(f" Loaded continual learning state")
            except Exception as e:
                if rank == 0:
                    print(f" Failed to load cl_state: {e}")
        
        if rank == 0:
            print(f"{'='*60}")
            print(f" Checkpoint loaded successfully!")
            print(f"{'='*60}\n")
    
    if rank == 0:
        print(f"\n[Task {current_task_id}] Re-configuring parameter gradients...")
    
    for param in trainer.model.model.parameters():
        param.requires_grad = False
    for param in trainer.model.lm_head.parameters():
        param.requires_grad = False
    if rank == 0:
        print("  Base model and lm_head frozen")
    
    if hasattr(trainer.model, 'scene_encoder'):
        for param in trainer.model.scene_encoder.parameters():
            param.requires_grad = True
        if rank == 0:
            print(" scene_encoder: trainable")
    else:
        if rank == 0:
            print(" WARNING: scene_encoder not found!")
    
    if hasattr(trainer.model.model, 'global_hypernet'):
        for param in trainer.model.model.global_hypernet.parameters():
            param.requires_grad = True
        if rank == 0:
            print(" global_hypernet: trainable")
    else:
        if rank == 0:
            print(" WARNING: global_hypernet not found!")

    trainable_count = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in trainer.model.parameters())
    if rank == 0:
        print(f" Trainable: {trainable_count:,} ({trainable_count/1e6:.2f}M)")
        print(f" Total: {total_count:,} ({total_count/1e6:.2f}M)")
        print(f" Ratio: {100*trainable_count/total_count:.4f}%")
    
    cl_manager.start_new_task(current_task_name)
    
    if rank == 0:
        print(f"\nStarting training for Task {current_task_id}: {current_task_name}")
    
    try:
        trainer.train() 
    except Exception as e:
        if rank == 0:
            print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    cl_manager.finish_task()
    
    if rank == 0:
        import os
        import torch
        import json
        
        save_path = os.path.join(training_args.output_dir, f'task_{current_task_id}')
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\n[Checkpoint] Saving ProtoStream components to {save_path}...")
        
        if hasattr(trainer.model, 'scene_encoder'):
            scene_path = os.path.join(save_path, 'scene_encoder.pt')
            
            scene_state = {}
            for name, param in trainer.model.scene_encoder.named_parameters():
                if param.requires_grad: 
                    scene_state[name] = param.data.cpu() 
            
            torch.save(scene_state, scene_path)
            size_mb = os.path.getsize(scene_path) / 1e6
            print(f" scene_encoder.pt ({size_mb:.1f}MB, {len(scene_state)} params)")

            if size_mb > 100:
                print(f"  WARNING: scene_encoder too large ({size_mb:.1f}MB)! Expected ~10MB")
                print(f"     This usually means vision_tower was saved by mistake!")
        
        if hasattr(trainer.model.model, 'global_hypernet'): 
            total_params = sum(p.numel() for p in trainer.model.model.global_hypernet.parameters())
            print(f"[DEBUG] global_hypernet total params: {total_params:,} ({total_params*4/1e6:.1f}MB)")
            
            for name, param in trainer.model.model.global_hypernet.named_parameters():
                print(f"  - {name}: {param.numel():,} ({param.numel()*4/1e6:.1f}MB)")
            hypernet_path = os.path.join(save_path, 'global_hypernet.pt')
            hypernet_state = trainer.model.model.global_hypernet.state_dict()
            hypernet_state = {k: v.cpu() for k, v in hypernet_state.items()}
            torch.save(hypernet_state, hypernet_path)
            
            size_mb = os.path.getsize(hypernet_path) / 1e6
            print(f"  global_hypernet.pt ({size_mb:.1f}MB)")
            
            if size_mb > 100:
                print(f" WARNING: global_hypernet too large ({size_mb:.1f}MB)! Expected ~30MB")
                print(f"     This usually means base_layer references were saved!")
        
        if hasattr(trainer.model, 'prototype_lib'):
            proto_path = os.path.join(save_path, 'prototype_lib.pt')
            
            proto_data = {
                'prototypes': torch.stack([p.data for p in trainer.model.prototype_lib.prototypes]) if len(trainer.model.prototype_lib.prototypes) > 0 else torch.empty(0, trainer.model.prototype_lib.proto_dim),
                'num_prototypes': len(trainer.model.prototype_lib.prototypes),
            }
            
            if hasattr(trainer.model.prototype_lib, 'task_ids'):
                proto_data['task_ids'] = trainer.model.prototype_lib.task_ids
            
            if hasattr(trainer.model.prototype_lib, 'frozen_mask'):
                proto_data['frozen_mask'] = trainer.model.prototype_lib.frozen_mask
            
            torch.save(proto_data, proto_path)
            
            size_mb = os.path.getsize(proto_path) / 1e6
            num_protos = proto_data['prototypes'].shape[0]
            print(f"prototype_lib.pt ({size_mb:.1f}MB, {num_protos} prototypes)")
            
            try:
                proto_stats = trainer.model.prototype_lib.get_statistics()
                if proto_stats:
                    serializable_stats = {}
                    for k, v in proto_stats.items():
                        if isinstance(v, (int, float, str, bool, list, dict)):
                            serializable_stats[k] = v
                        elif hasattr(v, 'item'):  # torch.Tensor with single value
                            serializable_stats[k] = v.item()
                    
                    with open(os.path.join(save_path, 'proto_stats.json'), 'w') as f:
                        json.dump(serializable_stats, f, indent=2)
            except Exception as e:
                print(f"Failed to save proto_stats: {e}")
        
        try:
            cl_state = {
                'current_task_id': current_task_id,
            }
            
            if hasattr(cl_manager, 'task_history'):
                cl_state['task_history'] = cl_manager.task_history
            
            torch.save(cl_state, os.path.join(save_path, 'cl_state.pt'))
        except Exception as e:
            print(f" Failed to save cl_state: {e}")
        
        total_size = sum(
            os.path.getsize(os.path.join(save_path, f))
            for f in os.listdir(save_path)
            if os.path.isfile(os.path.join(save_path, f))
        )
        print(f" Total checkpoint size: {total_size/1e6:.1f}MB")
    
    if dist.is_initialized():
        dist.barrier()
    
    return trainer

def train(attn_implementation=None):
    global local_rank
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )
    # import ipdb; ipdb.set_trace()
    model = get_model(model_args, training_args, data_args, bnb_model_from_pretrained_args)
    model.config.use_cache = False
    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if "mistral" in model_args.model_name_or_path.lower() or "mixtral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="left")
    elif "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")
    elif (
        "wizardlm-2" in model_args.model_name_or_path.lower()
        or "vicuna" in model_args.model_name_or_path.lower()
        or "llama" in model_args.model_name_or_path.lower()
        or "yi" in model_args.model_name_or_path.lower()
        or "nous-hermes" in model_args.model_name_or_path.lower()
        and "wizard-2" in model_args.model_name_or_path.lower()
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    rank0_print(f"Prompt version: {model_args.version}")
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_grid_pinpoints is not None:
            if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
                try:
                    patch_size = data_args.image_processor.size[0]
                except Exception as e:
                    patch_size = data_args.image_processor.size["shortest_edge"]

                assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
                # Use regex to extract the range from the input string
                matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
                range_start = tuple(map(int, matches[0]))
                range_end = tuple(map(int, matches[-1]))
                # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
                grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
                # Multiply all elements by patch_size
                data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
            elif isinstance(data_args.image_grid_pinpoints, str):
                data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_crop_resolution = data_args.image_crop_resolution
        model.config.image_split_resolution = data_args.image_split_resolution
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_newline_position = model_args.mm_newline_position
        model.config.add_faster_video = model_args.add_faster_video
        model.config.faster_token_stride = model_args.faster_token_stride
        model.config.force_sample = data_args.force_sample
        model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride 

        ### Deciding train which part of the model
        if model_args.mm_tunable_parts is None:  # traditional way of deciding which part to train
            if training_args.use_protostream:
                rank0_print("[ProtoStream] Freezing entire model for ProtoStream")
                model.requires_grad_(False)
                vision_tower.requires_grad_(False)
                model.get_model().mm_projector.requires_grad_(False)
                model.get_model().vision_resampler.requires_grad_(False)
            else:
                model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
                model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
                if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
                    model.requires_grad_(False)
                if model_args.tune_mm_mlp_adapter:
                    for p in model.get_model().mm_projector.parameters():
                        p.requires_grad = True
                if model_args.tune_mm_vision_resampler:
                    for p in model.get_model().vision_resampler.parameters():
                        p.requires_grad = True

                model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
                if training_args.freeze_mm_mlp_adapter:
                    for p in model.get_model().mm_projector.parameters():
                        p.requires_grad = False

                model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
                if training_args.freeze_mm_vision_resampler:
                    for p in model.get_model().vision_resampler.parameters():
                        p.requires_grad = False

                model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
                if model_args.unfreeze_mm_vision_tower:
                    vision_tower.requires_grad_(True)
                else:
                    vision_tower.requires_grad_(False)

        else:
            rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
            model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
            # Set the entire model to not require gradients by default
            model.requires_grad_(False)
            vision_tower.requires_grad_(False)
            model.get_model().mm_projector.requires_grad_(False)
            model.get_model().vision_resampler.requires_grad_(False)
            # Parse the mm_tunable_parts to decide which parts to unfreeze
            tunable_parts = model_args.mm_tunable_parts.split(",")
            if "mm_mlp_adapter" in tunable_parts:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if "mm_vision_resampler" in tunable_parts and training_args.token_compression=="resampler":
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True
            if "mm_vision_tower" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" in name:
                        param.requires_grad_(True)
            # if "mm_language_model" in tunable_parts:
            #     for name, param in model.named_parameters():
            #         if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
            #             param.requires_grad_(True)
            if "mm_language_model" in tunable_parts:
                if training_args.use_lora:
                    rank0_print("Using LoRA for language model fine-tuning - keeping base model frozen")
                    pass  
                else:
                    rank0_print("Full parameter fine-tuning for language model")
                    for name, param in model.named_parameters():
                        if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
                            param.requires_grad_(True)
            
            if "mm_lora_layer" in tunable_parts:
                for name, param in model.named_parameters():
                    if "lora" in name:
                        param.requires_grad_(True)
        
        for name, param in model.named_parameters():
            if param.requires_grad:  # Check if the parameter requires training
                rank0_print(name)
        total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
        trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
        rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if data_args.data_augmentation:
        data_args.transform_train = v2.Compose([
            v2.ToImage(),
            v2.ColorJitter(brightness=0.2, saturation=0.2),
            v2.RandomPosterize(bits=4),
            v2.RandomAdjustSharpness(sharpness_factor=1.5),
            v2.RandomAutocontrast(),
            v2.ToPILImage()
        ])
    else:
        data_args.transform_train = None

    # import ipdb; ipdb.set_trace()
    data_module = make_supervised_data_module(tokenizer=tokenizer,vision_tower=vision_tower, data_args=data_args)
    
    params_no_grad = [
        n for n, p in model.named_parameters() if not p.requires_grad
    ]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print(
                    '[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'
                    .format(len(params_no_grad), params_no_grad))
            else:
                print(
                    '[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'
                    .format(len(params_no_grad),
                            ', '.join(params_no_grad[:10])))
            print(
                "[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental."
            )
            print(
                "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining"
            )
            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('ignored_parameters', True)
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args,
                                **kwargs,
                                use_orig_params=use_orig_params)
                return wrap_func
            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)
    
    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    # print(list(model.get_model().vision_resampler.parameters())[0])
    # import ipdb; ipdb.set_trace()

    if training_args.continual_learning and training_args.use_tucker_4d:
        rank0_print("=" * 50)
        rank0_print("Starting Continual Learning with 4D Tucker-LoRA")
        rank0_print(f"Task ID: {training_args.current_task_id}")
        rank0_print(f"Total Tasks: {training_args.num_tasks}")
        rank0_print(f"Scene Num: {training_args.tucker_scene_num}")
        rank0_print(f"Env Num: {training_args.tucker_env_num}")
        rank0_print("=" * 50)
        trainer = train_with_continual_learning(trainer, training_args, data_module)
    elif training_args.continual_learning and training_args.use_protostream:
        rank0_print("=" * 50)
        rank0_print("Starting Continual Learning with ProtoStream")
        rank0_print(f"Total Tasks: {training_args.num_tasks}")
        rank0_print(f"Proto Dim: {training_args.proto_dim}")
        rank0_print(f"Similarity Threshold: {training_args.similarity_threshold}")
        rank0_print(f"Max Prototypes: {training_args.max_prototypes}")
        rank0_print("=" * 50)

        rank0_print("[ProtoStream] Force freezing base model and lm_head...")
        for param in trainer.model.model.parameters():
            param.requires_grad = False
        for param in trainer.model.lm_head.parameters():
            param.requires_grad = False
        rank0_print("[ProtoStream] Base model frozen ✓")

        rank0_print("[ProtoStream] Ensuring ProtoStream components are trainable...")
        
        if hasattr(trainer.model, 'scene_encoder'):
            for param in trainer.model.scene_encoder.parameters():
                param.requires_grad = True
            rank0_print(" scene_encoder: trainable")
        else:
            rank0_print("WARNING: scene_encoder not found!")
        
        if hasattr(trainer.model.model, 'global_hypernet'):
            for param in trainer.model.model.global_hypernet.parameters():
                param.requires_grad = True
            rank0_print("global_hypernet: trainable")
        else:
            rank0_print("WARNING: global_hypernet not found!")
        
        rank0_print("[ProtoStream] ProtoStream components ready ✓")

        trainable_params = []
        frozen_params = []
        
        for name, param in trainer.model.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param.numel()))
            else:
                frozen_params.append((name, param.numel()))
        
        total_trainable = sum(p[1] for p in trainable_params)
        total_frozen = sum(p[1] for p in frozen_params)
        
        rank0_print(f"Trainable parameters: {total_trainable:,} ({total_trainable/1e6:.2f}M)")
        rank0_print(f"Frozen parameters: {total_frozen:,} ({total_frozen/1e6:.2f}M)")
        rank0_print(f"Trainable ratio: {100*total_trainable/(total_trainable+total_frozen):.2f}%")
        
        
        rank0_print("\nTrainable parameters:")
        for name, numel in trainable_params[:10]:
            rank0_print(f"  - {name}: {numel:,}")
        if len(trainable_params) > 10:
            rank0_print(f"  ... and {len(trainable_params)-10} more")
        
        from streamvln.model.triple_distillation import TripleDistillationLoss
        from streamvln.model.continual_learning import ProtoStreamCL
        
        distiller = TripleDistillationLoss(
            model=trainer.model,
            lambda_sp=training_args.lambda_sp,
            lambda_pp=training_args.lambda_pp,
            lambda_cp=training_args.lambda_cp,
            temperature=training_args.distill_temperature
        )
        
        cl_manager = ProtoStreamCL(
            model=trainer.model,
            distiller=distiller
        )
    
        current_task_id = training_args.current_task_id
        current_task_name = task_list[current_task_id] if current_task_id < len(task_list) else f"task_{current_task_id}"

        rank0_print(f"\n{'='*60}")
        rank0_print(f"Task {current_task_id+1}/{len(task_list)}: {current_task_name}")
        rank0_print(f"{'='*60}")

        trainer = train_protostream(
            trainer=trainer,
            training_args=training_args,
            data_module=data_module,
            cl_manager=cl_manager,
            current_task_id=current_task_id,
            current_task_name=current_task_name
        )
    else:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        # import ipdb; ipdb.set_trace()
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        if training_args.fsdp:
            safe_save_model_for_hf_trainer_fsdp(trainer=trainer, output_dir=training_args.output_dir, model_args=model_args)
        else:
            safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, model_args=model_args)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()