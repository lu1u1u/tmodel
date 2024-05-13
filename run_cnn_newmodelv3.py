#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
from cmath import log
import numpy as np
import logging
import math
import os
import sys
from tqdm import tqdm
import pickle
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
from copy import deepcopy
import datasets
import evaluate
import torch
from datasets import load_dataset, Dataset, DatasetDict

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from newtv3 import NewTModel

os.environ["WANDB_DISABLED"] = "true"

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.28.0.dev0")

require_version("datasets>=1.8.0",
                "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default='gpt2',
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    """
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )
    """
    # added 0401
    ztokens: Optional[int] = field(default=32)  # ztoken 的长度
    maxztokens: Optional[int] = field(default=64)  # ztokens 最长
    shallow_decoder_n_layer: Optional[int] = field(default=6)
    zdim: Optional[int] = field(default=32)
    znorm: Optional[int] = field(default=0)
    alpha: Optional[float] = field(default=1.0)
    beta: Optional[float] = field(default=1.0)
    large_path: Optional[str] = field(default='gpt2-large')
    cnnf: str = field(default=None)
    use_flash_attention: Optional[bool] = field(default=False)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default='dummy.txt', metadata={
                                      "help": "The input training data file (a text file)."})

    validation_file: Optional[str] = field(
        default='dummy.txt',
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={
                            "help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    # added
    encstr: str = field(
        default='suffix',
        metadata={"help": "suffix or all"},
    )
    ptae: str = field(
        default=None,
        metadata={"help": "suffix or all"},
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0",
                            "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_flash_attention:
        check_min_version("4.35.0")
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokenizer.add_special_tokens({'sep_token': '<SEP>'})

    special_list = [f'<THO{idx}>' for idx in range(model_args.ztokens)]
    special_seq = ''.join(special_list)
    # print(special_list)
    # print(special_seq)
    tokenizer.add_special_tokens({'additional_special_tokens': special_list})
    # print(tokenizer)
    
    tholist = [tokenizer.convert_tokens_to_ids(
                f'<THO{i}>') for i in range(model_args.ztokens)]

    thoid = tokenizer.convert_tokens_to_ids('<THO0>')
    sepid = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    print("pad id", tokenizer.convert_tokens_to_ids("<PAD>"))
    print("thoid id", thoid)
    assert thoid > sepid

    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        streaming=data_args.streaming,
    )

    """
    validation_dataset = raw_datasets['validation']
    num_samples = len(validation_dataset)
    validation_subset = validation_dataset.select(range(100))
    raw_datasets = DatasetDict()
    raw_datasets['validation'] = validation_subset
    """

    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base")
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    
    # should do this due to past_key_values limitation
    block_size -= model_args.ztokens

    enc_block_size = 512

    def safe_encode(srclist, tarlist, tail_trunc_length=512):
        # ae-encoder
        encres = {
            'input_ids': [],
            'attention_mask': []
        }
        ae_decres = {
            'input_ids': [],
            'attention_mask': []
        }

        # main decoder
        decres = {
            'input_ids': [],
            'attention_mask': [],
        }

        for src, tar in zip(srclist, tarlist):
            # generating full encstrs:
            enctail = tar
            enctail_ids = tokenizer.encode(enctail)

            if data_args.encstr == "all":
                exit("not ready")
                if len(enctail_ids) >= block_size // 1.5:  # ill tail
                    enctail_ids = enctail_ids[:tail_trunc_length]

                enchead = src + sep_token

                enchead_ids = tokenizer.encode(
                    enchead)[:block_size-len(enctail_ids)]

                encres_app = enchead_ids + enctail_ids + \
                    [tokenizer.pad_token_id] * \
                    max(block_size-len(enchead_ids)-len(enctail_ids), 0)
            else:
                assert data_args.encstr == "suffix"

                # if len(enctail_ids) >= block_size // 1.5: # ill tail
                enctail_ids = enctail_ids[:enc_block_size-model_args.ztokens]+tholist.copy()

                encres_app = enctail_ids + \
                    [tokenizer.pad_token_id] * \
                    max(enc_block_size-len(enctail_ids), 0)

            encres_atm = [
                1 if i != tokenizer.pad_token_id else 0 for i in encres_app]

            assert len(encres_app) == len(encres_atm) == enc_block_size

            encres["input_ids"].append(encres_app)
            encres["attention_mask"].append(encres_atm)


            # generating  ae-decoder:
            ae_dec_ids = tokenizer.encode(tar)
            ae_dec_ids = ae_dec_ids[:enc_block_size]

            ae_dec_app = ae_dec_ids + \
                    [tokenizer.pad_token_id] * \
                    max(enc_block_size-len(ae_dec_ids), 0)
            
            ae_decres["input_ids"].append(ae_dec_app)
            ae_dec_atm = [
                1 if i != tokenizer.pad_token_id else 0 for i in ae_dec_app]
            ae_decres["attention_mask"].append(ae_dec_atm)

            assert len(ae_dec_app) == len(ae_dec_atm) == enc_block_size

            # generating  decstrs:
            dectail = tokenizer.sep_token + special_seq + tar + tokenizer.eos_token
            dectail_ids = tokenizer.encode(dectail)

            if len(dectail_ids) >= block_size // 1.5:  # ill tail
                dectail_ids = dectail_ids[:tail_trunc_length]

            dechead = src
            dechead_ids = tokenizer.encode(dechead)[:block_size-len(dectail_ids)]

            decres_app = dechead_ids + dectail_ids + \
                [tokenizer.pad_token_id] * \
                max(block_size-len(dechead_ids)-len(dectail_ids), 0)
            decres_atm = [
                1 if i != tokenizer.pad_token_id else 0 for i in decres_app]
            
            assert len(decres_app) == len(decres_atm) == block_size

            decres["input_ids"].append(decres_app)
            decres["attention_mask"].append(decres_atm)

            # decres["labels"].append(decres_label)

        return encres,ae_decres, decres

    def tokenize_function(examples, suffix_loss):
        with CaptureLogger(tok_logger) as cl:
            bs = len(examples['article'])

            encres,ae_decres, decres = safe_encode(
                srclist=examples['article'], tarlist=examples['highlights'])

            # print()
            # print(encres["input_ids"][:2])
            # print()
            # print(ae_decres["input_ids"][:2])
            # print()
            # print(decres["input_ids"][:2])
            # exit()

            input_ids = decres['input_ids']
            attention_mask = decres['attention_mask']

            labels = input_ids.copy()

            for i in range(bs):
                start_pos = input_ids[i].index(thoid)

                # for j in range(start_pos, start_pos + model_args.ztokens):
                    # pos_mask[i][j] = True

                # print(labels[i])
                pad_label = [-100 if label_token in tholist or label_token == tokenizer.pad_token_id or label_token == sepid else label_token
                             for label_token in labels[i]]

                if suffix_loss:
                    pad_label[:start_pos] = [-100] * len(pad_label[:start_pos])
                    # print(pad_label)
                    # exit()

                labels[i] = pad_label


            input_ids_enc = encres['input_ids']
            attention_mask_enc = encres['attention_mask']


            input_ids_ae_dec = ae_decres['input_ids']
            attention_mask_ae_dec = ae_decres['attention_mask']
            labels_ae = [
                [-100 if i in tholist or i ==
                    tokenizer.pad_token_id or i == sepid else i for i in j]
                for j in input_ids_ae_dec
            ]

        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,

            'input_ids_enc': input_ids_enc,
            'attention_mask_enc': attention_mask_enc,

            'input_ids_ae_dec': input_ids_ae_dec,
            'attention_mask_ae_dec': attention_mask_ae_dec,

            'labels_ae': labels_ae
        }

    lm_datasets = DatasetDict()

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            # lm_datasets = raw_datasets.map(
            #     tokenize_function,
            #     batched=True,
            #     num_proc=data_args.preprocessing_num_workers,
            #     remove_columns=column_names,
            #     load_from_cache_file=not data_args.overwrite_cache,
            #     desc="Running tokenizer on dataset",
            # )

            # added
            lm_datasets["validation"] = raw_datasets["validation"].map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
                fn_kwargs={"suffix_loss": True}
            )
            lm_datasets["train"] = raw_datasets["train"].map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
                fn_kwargs={"suffix_loss": False}
            )

        else:
            exit("no streaming")
            lm_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    from peft import get_peft_model,LoraConfig

    lora_config = LoraConfig(
        r=128,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    z_start_id = tokenizer.convert_tokens_to_ids('<THO0>')

    config.update({"ztokens": model_args.ztokens, "zdim": model_args.zdim, 
                    "z_start_id":z_start_id, "len_tokenizer":len(tokenizer)})

    if data_args.ptae is not None:
        logger.info(f"loading pretrained ae model {data_args.ptae}")
        tmodel = NewTModel.from_pretrained(data_args.ptae, config=config, model_args=model_args, lora_config=lora_config)
    else:
        tmodel = NewTModel(config=config, model_args=model_args, lora_config=lora_config)
    
    tmodel.freeze()

    trainable_parameters = 0
    all_param = 0
    for pname, param in tmodel.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            logger.info(pname)
            trainable_parameters += param.numel()
    
    logger.info(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")
    

    # print(model_args.ztokens, tokenizer.model_max_length)
    # print(lm_datasets)

    """
    try:     
        print(tokenizer.batch_decode(
            lm_datasets['train']["input_ids"][:5],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ))
        print(tokenizer.batch_decode(
            lm_datasets['train']["input_ids_enc"][:5],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ))
        print(tokenizer.batch_decode(
            lm_datasets['train']["input_ids_enc_z"][:5],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ))
    except:
        print("ERROR during decoding...")
    """

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        def calc_acc(preds, labels):
            totals = 0
            exm = 0
            for idx, i in enumerate(labels):
                if i.item() != -100:
                    totals += 1
                    if i.item() == preds[idx].item():
                        exm += 1
            res = exm / totals if totals else 0
            return {'accuracy': res}

        def compute_metrics(eval_preds):

            preds, labels = eval_preds
            # print(labels)
            labels = labels[0]
            # print(type(labels))
            for idx, i in enumerate(labels):
                pos = 0
                while pos < len(i):
                    if i[pos] == tokenizer.sep_token_id:
                        labels[idx][pos] = -100
                        break
                    else:
                        labels[idx][pos] = -100
                        pos += 1

            # torch.set_printoptions(profile='full', precision=1)
            # print(torch.tensor(labels))

            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels

            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return calc_acc(preds, labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=tmodel,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_train:
        kwargs = {"finetuned_from": model_args.model_name_or_path,
                  "tasks": "text-generation"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
