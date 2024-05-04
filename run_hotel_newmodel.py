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
from newt import NewTModel
from mytrainer import MyTrainer
os.environ["WANDB_DISABLED"] = "true"

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.28.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

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
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
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
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
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
    ztokens: Optional[int] = field(default = 32) # ztoken 的长度
    maxztokens: Optional[int] = field(default = 64) # ztokens 最长
    shallow_decoder_n_layer : Optional[int] = field(default = 6) 
    zdim : Optional[int] = field(default = 32) 
    alpha : Optional[int] = field(default = 1) 
    beta : Optional[int] = field(default = 1) 
    large_path : Optional[str] = field(default = 'gpt2-large') 
    
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
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    
    
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
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
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
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

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
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
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
      
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    tokenizer.add_special_tokens({'pad_token' : '<PAD>'})
    special_list = [f'<THO{idx}>' for idx in range(model_args.ztokens)]
    special_seq = ''.join(special_list)
    print(special_list)
    print(special_seq)
    tokenizer.add_special_tokens({'additional_special_tokens':special_list})
    print(tokenizer)
    
    


    def postprocess(examples, default_spt = '.'):

        all_encstr = []
        all_decstr = []
        
        
        for str in examples['text']:
                
            # first add bos and eos
            str = tokenizer.bos_token + str + tokenizer.eos_token
            # handle split
            split_token = default_spt
            choices = [default_spt, ',', '!', '?', ';', '，', '。', '、', '！','？','END']
               
            for s in choices:
                if s == 'END':
                    split_token = ' '
                    decstr = str.split()
                    break
                else:
                    split_token = s
                    decstr = str.split(split_token)
                if len(decstr) == 1:
                    continue
                else:
                    break
            
            if len(decstr) == 1:
                print(decstr,'Unexpected occurred')
                decstr = list(decstr[0])
                
            where_to_add = min(math.floor(len(decstr) / 4),1) 

            encstr = split_token.join(decstr[where_to_add+1:])
        
            decstr[where_to_add+1] = special_seq + decstr[where_to_add+1]
            decstr = split_token.join(decstr)

            all_encstr.append(encstr)
            all_decstr.append(decstr)
            
        return{
            'encstr' : all_encstr,
            'decstr' : all_decstr,
            'special_seq' : [special_seq for _ in range(len(examples['text']))]
        }
        
        
    
    raw_datasets = DatasetDict() 
    print(data_args.train_file)

    all_data = load_dataset(
        'csv', 
        data_files=data_args.train_file, 
        split='train'
    )
    remove_feas = [fea for fea in all_data.features if fea != 'text']
    all_data = all_data.remove_columns(remove_feas)
    print(all_data,len(all_data))
    
    def tklen(s):
        return len(tokenizer.encode(s))
    good_list = [{'text' : i} for i in tqdm(all_data['text'], desc='checking length') if 20 <= tklen(i) <= 512]
    total_len = len(good_list)
    
    train_range = math.floor(total_len*0.96)
    val_range = math.floor(total_len*0.98)
    test_range = total_len
    
    trains = Dataset.from_list(good_list[:train_range])
    vals = Dataset.from_list(good_list[train_range:val_range])
    
    
    raw_datasets['train'] = trains
    raw_datasets['validation'] = vals
    
    
   
    raw_datasets = raw_datasets.map(
        postprocess,
        remove_columns='text',
        batched=True,
        num_proc=8
    )       
    print(raw_datasets)

    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
 
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
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
    
    # modified for large blocksize
    if model_args.ztokens > 32:
        block_size = 256
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            thoid = tokenizer.convert_tokens_to_ids('<THO0>')
            tholist = [tokenizer.convert_tokens_to_ids(f'<THO{i}>') for i in range(model_args.ztokens)]
            encres = tokenizer(
                examples['encstr'], 
                padding='max_length', 
                max_length = block_size - model_args.ztokens,
                truncation = True
            )
            decres = tokenizer(
                examples['decstr'], 
                padding='max_length', 
                max_length = block_size,
                truncation = True
            )
            zres = tokenizer(
                examples['special_seq']
            )
            
            input_ids = decres['input_ids']
            attention_mask = decres['attention_mask']
            input_ids_z = zres['input_ids']
            attention_mask_z = zres['attention_mask']
            
            pos_mask = torch.zeros((len(input_ids), block_size), dtype=torch.bool)
            for i in range(pos_mask.size(0)):
                try:
                    start_pos = input_ids[i].index(thoid)
                except:
                    with open("./wrongs.txt", "a") as f:
                        f.write("==========================\n")
                        f.write(examples['decstr'][i] + '\n')
                        f.write(tokenizer.decode(input_ids[i]) + '\n')
                    assert 0
                for j in range(start_pos, start_pos + model_args.ztokens):  
                    pos_mask[i][j] = True
                    
            labels = [
                [-100 if i in tholist or i == tokenizer.pad_token_id else i for i in j] 
                for j in input_ids
            ]
            
            input_ids_enc = encres['input_ids']
            attention_mask_enc = encres['attention_mask']
            
            labels_enc = [
                [-100 if i in tholist or i == tokenizer.pad_token_id else i for i in j] 
                for j in input_ids_enc
            ]

            

        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
            
        return {
            'input_ids' : input_ids,
            'input_ids_enc' : input_ids_enc,
            'input_ids_enc_z' : input_ids_z,
            'attention_mask' : attention_mask,
            'attention_mask_enc' : attention_mask_enc,
            'attention_mask_enc_z' : attention_mask_z,
            'labels' : labels,
            'pos_mask' : pos_mask,
            'labels_enc' : labels_enc
        }

    
    
    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            lm_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            lm_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    
    tmodel = NewTModel(config = config, model_args = model_args )
    tmodel.build_ed(len(tokenizer))
    
    print(model_args.ztokens, tokenizer.model_max_length) 
    print(lm_datasets)
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


    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        #metric = evaluate.load("accuracy")
        
        def calc_acc(preds, labels):
            totals = 0
            exm = 0
            for idx, i in enumerate(labels):
                if i.item() != -100:
                    totals += 1
                    if i.item() == preds[idx].item():
                        exm += 1
            res =  exm / totals if totals else 0      
            return {'accuracy': res} 
        
        def compute_metrics(eval_preds):
            
            
            preds, labels = eval_preds
            labels = labels[0]
            
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return calc_acc(preds, labels)
        
        def postprocess_text(preds, labels):
            preds = [pred.strip() if pred.strip() else ' ' for pred in preds]
            labels_sg = [label.strip() for label in labels]
            labels = [[label.strip()] for label in labels]
            return preds, labels, labels_sg
        
        def get_self_belu_score(corpus: list) -> float:
            # calculate self-BELU score on corpus
            score = 0.0
            cnt = 0
            length = len(corpus)
            for index in range(length):
                curr_text = [corpus[index]]    
                other_text = [corpus[:index] + corpus[index + 1:]]   
                curr_belu_score = bleu_metric.compute(predictions=curr_text, references=other_text)  # float
                score += curr_belu_score['bleu']
                cnt += 1
            return score / cnt if cnt else 0
        
        ppl_metric = evaluate.load("perplexity")
        bleu_metric = evaluate.load("bleu")
        sacrebleu_metric = evaluate.load("sacrebleu")
        rouge_metric = evaluate.load("rouge")
        selfbleu_metric = get_self_belu_score
        gpt2_large_path = "/yinyongjing/hfmodels/gpt2/large"
        
        def compute_metrics_afterall(eval_preds):
            preds, labels = eval_preds
            labels = labels[0]

            labels = labels[:, 1:]
            preds = preds[:, :-1]
            
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds, decoded_labels, decoded_labels_sg = postprocess_text(decoded_preds, decoded_labels)

            result = {}
            logger.info("calculating ppl...")
            ppl = ppl_metric.compute(predictions=decoded_preds, model_id=gpt2_large_path)
                
            logger.info("calculating bl...")
            bl = sacrebleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
            
            #logger.info("calculating sbl...")
            #try:
            #    sbl = selfbleu_metric(corpus=decoded_preds)
            #except:
            #    logger.info("error while calc. sbl...")
            #    sbl = 0.0
                
            logger.info("calculating rouge...")
            rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels_sg)
            
            #print(ppl['mean_perplexity'])
            #print(bl['score'])
            #print(sbl)
            #print(rouge['rougeL'])
            #print(rouge['rougeLsum'])
            
            result['PPL'] = ppl['mean_perplexity']
            result['BL'] = bl['score']
            #result['S-BL'] = sbl
            result['R-L'] = rouge['rougeL']
            result['R-Lsum'] = rouge['rougeLsum']
            result['Len'] = np.mean([len(s) for s in decoded_preds])
            
            result = {k: round(v, 4) for k, v in result.items()}
            return result

    # Initialize our Trainer
    trainer = Trainer(
        model=tmodel,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
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
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
       
        
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        
    if training_args.do_train:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
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
