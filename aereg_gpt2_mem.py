from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers import (
    AutoModel,
    PreTrainedModel,
    AutoModelForCausalLM
)
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
import math
import os
from re import S
import warnings
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import random


import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Model
# from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block

from modeling_gpt2 import GPT2Attention, GPT2Block, GPT2LMHeadModel, GPT2Model


from transformers.activations import ACT2FN
from transformers.utils import ModelOutput
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.modeling_outputs import (
    CausalLMOutput,
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)


@dataclass
class AEOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_z: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


logger = logging.get_logger(__name__)


class PrefixEncoder(nn.Module):
    def __init__(self, config, model_args):
        super().__init__()

        self.input_dim = config.zdim
        self.hidden_dim = config.n_embd

        self.prefix_seq_len = config.ztokens
        self.match_n_layer = model_args.shallow_decoder_n_layer

        self.prefix_mlp = nn.Linear(
            self.input_dim, self.match_n_layer * 2 * config.n_embd)

        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head

    def forward(
        self,
        input_embd
    ):

        batch_size = input_embd.size(0)

        past_key_values = self.prefix_mlp(input_embd)

        past_key_values = past_key_values.view(
            batch_size, self.prefix_seq_len, self.match_n_layer, -1)

        # Resize
        past_key_values = past_key_values.view(
            batch_size,
            self.prefix_seq_len,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )

        # Transpose -> [match_n_layer*2, batch_size, match_n_head, prefix_seq_len, match_n_embd]
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4])
        past_key_values = torch.split(past_key_values, 2)

        all_kvs = ()
        for i in range(len(past_key_values)):
            kvpair = (past_key_values[i][0], past_key_values[i][1])
            all_kvs += (kvpair,)
        # print("finally:")
        # print(all_kvs[0][0])
        # print(f"len all_kvs: {type(all_kvs), len(all_kvs)}")
        # print(f"len all_kvs[0]: {type(all_kvs[0]),len(all_kvs[0])}")
        # print(f"shape of k/v : {type(all_kvs[0][0]), all_kvs[0][0].shape, all_kvs[0][1].shape}")
        # assert 0

        return all_kvs


from peft import get_peft_model

from modeling_gpt2 import GPT2LMHeadModel, GPT2Model

class AE(PreTrainedModel):
    _supports_flash_attn_2 = True

    def __init__(self, config, model_args, lora_config=None):
        super().__init__(config)
        self.ztokens = config.ztokens
        
        self.zwte = nn.Embedding(self.ztokens, config.n_embd)

        if model_args.from_scratch:
            self.encoder = GPT2Model.from_pretrained(
                model_args.model_name_or_path,
                config=config,
            )

            # self.encoder = GPT2Model(config.update({"_attn_implementation":"flash_attention_2"}))
            # self.decoder = GPT2LMHeadModel(config.update({"_attn_implementation":"flash_attention_2"}))
            # self.encoder = GPT2Model(config)
            dec_config = deepcopy(config)
            dec_config.max_position_embeddings += self.ztokens
            dec_config.n_layer = model_args.shallow_decoder_n_layer
            self.decoder = GPT2LMHeadModel(dec_config)
        else:
            raise NotImplementedError

            self.encoder = AutoModelForCausalLM.from_pretrained(
                model_args.ae_model_name_or_path,
                config=config,
                use_flash_attention_2=True
            )
            self.decoder = AutoModelForCausalLM.from_pretrained(
                model_args.ae_model_name_or_path,
                config=config,
                use_flash_attention_2=True
        )

        # self.encoder.resize_token_embeddings(config.len_tokenizer)
        self.decoder.resize_token_embeddings(config.len_tokenizer)

        if lora_config is not None:
            self.encoder = get_peft_model(self.encoder, lora_config)

        block_config = deepcopy(self.config)

        block_config.add_cross_attention = True
        block_config._attn_implementation = "eager"

        self.cross_block = GPT2Block(block_config)

        self.lnz = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.prefix_encoder = PrefixEncoder(config, model_args)

        self.proj = nn.Linear(config.hidden_size, config.zdim, bias=False)
        self.up = nn.Linear(config.zdim, config.hidden_size, bias=False)

        self.z_start_id = config.z_start_id
        self.z_end_id = self.z_start_id + self.ztokens

        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))


    def forward(
        self,
        input_ids_enc,
        attention_mask_enc,
        input_ids_dec,
        attention_mask_dec,
        labels,
    ):
        with torch.no_grad():
            enc_outs = self.encoder(
                input_ids=input_ids_enc,
                attention_mask=attention_mask_enc
            )

        bz = input_ids_enc.size(0)
        z_idx = torch.arange(0, self.ztokens, dtype=torch.long, device=input_ids_enc.device)

        input_ids_enc_z = z_idx.unsqueeze(0).repeat(bz, 1)
        hidden_states_z = self.zwte(input_ids_enc_z)

        attention_mask_enc_4d = attention_mask_enc.view(
            input_ids_enc.size(0), -1)
        attention_mask_enc_4d = attention_mask_enc_4d[:, None, None, :]

        attention_mask_enc_4d = attention_mask_enc_4d.to(dtype=self.dtype)
        attention_mask_enc_4d = (
            1.0 - attention_mask_enc_4d) * torch.finfo(self.dtype).min

        enc_lhs = enc_outs.last_hidden_state

        hidden_z = self.cross_block(
            hidden_states_z,
            encoder_attention_mask = attention_mask_enc_4d,
            encoder_hidden_states = enc_lhs
        )[0]

        hidden_z = self.lnz(hidden_z)
        hidden_down = self.proj(hidden_z)

        # add noise?
        loss = None
        if labels is not None:
            past_key_values = self.prefix_encoder(hidden_down)

            attention_mask_z = attention_mask_dec.new_ones(bz, hidden_down.size(1))
            
            dec_outs = self.decoder.transformer(
                input_ids=input_ids_dec,
                past_key_values=past_key_values,
                attention_mask=torch.cat(
                    (attention_mask_z, attention_mask_dec), dim=-1),
                output_hidden_states=True,
                output_attentions=False
            )
            
            lhs = dec_outs.last_hidden_state

            lm_logits = self.decoder.lm_head(lhs)

            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return AEOutput(
            loss=loss,
            hidden_z=hidden_down
        )


class NewTModel(PreTrainedModel):
    _supports_flash_attn_2 = True

    def __init__(self, config, model_args, lora_config=None):
        super().__init__(config)

        self.ztokens = config.ztokens
        
        self.z_start_id = config.z_start_id
        self.z_end_id = self.z_start_id + self.ztokens

        self.alpha = model_args.alpha
        self.beta = model_args.beta

        self.model_args = model_args

        # self.mseloss = F.smooth_l1_loss
        self.mseloss = nn.MSELoss()
        self.znorm = model_args.znorm

        self.model_parallel = False
        self.device_map = None
        
        if model_args.from_scratch:
            # self.main_decoder = GPT2LMHeadModel(config.update({"_attn_implementation":"flash_attention_2"}))
            
            # max_position_embeddings
            dec_config = deepcopy(config)
            dec_config.max_position_embeddings += self.ztokens

            self.main_decoder = GPT2LMHeadModel(dec_config)
        else:
            raise NotImplementedError
            self.main_decoder = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                config=config,
                use_flash_attention_2=True
            )

        self.proj = nn.Linear(dec_config.hidden_size, dec_config.zdim, bias=False)
        self.proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        if self.beta > 0 or self.alpha > 0:
            self.aemodel = AE(deepcopy(config), model_args=model_args, lora_config=lora_config)

        # will revise config.vocab_size
        self.main_decoder.resize_token_embeddings(dec_config.len_tokenizer)

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.balanceMse = 1
        self.mseclock = 0

        self.balanceRec = 1
        self.recclock = 0
        self.recthr = 2.00

        self.balanceNll = 1
        self.nllclock = 0
        self.nllthr = 2.40

        self.trigger_time = 100

        self.post_init()


    def freeze(self):
        if hasattr(self, "aemodel"):
            self.aemodel.decoder.requires_grad_(False)

    def load_encoder_and_fix(self, model_name_or_path, config):
        if hasattr(self, "aemodel"):
            self.aemodel.encoder = GPT2Model.from_pretrained(
                model_name_or_path,
                config=config,
                )
            # self.aemodel.encoder.resize_token_embeddings(config.len_tokenizer)
            self.aemodel.encoder.requires_grad_(False)

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.main_decoder.h),
                           range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.aemodel.transformer.h))
        self.main_decoder.parallelize(self.device_map)
        self.aemodel.transformer.parallelize(self.device_map)
        self.aemodel.decoder.parallelize(self.device_map)
        self.first_device = self.main_decoder.first_device
        self.model_parallel = True

    # DELLA
    def deparallelize(self):
        self.aemodel.encoder.deparallelize()
        self.aemodel.decoder.deparallelize()
        self.main_decoder.deparallelize()
        torch.cuda.empty_cache()

    def resize_token_embeddings(self, len_t):
        self.main_decoder.resize_token_embeddings(len_t)

    def generate(self, *args, **kwargs):
        return self.main_decoder.generate(*args, **kwargs)

    def balance_trigger(self, mseloss, recloss, nllloss):
        return
        if recloss <= self.recthr:
            self.recclock += 1
        else:
            self.recclock -= 1
        if self.recclock <= 0:
            self.balanceRec = 1
        elif self.recclock >= self.trigger_time:
            self.balanceRec = 0

        if nllloss <= self.nllthr:
            self.nllclock += 1
        else:
            self.nllclock -= 1
        if self.nllclock <= 0:
            self.balanceNll = 1
        elif self.nllclock >= self.trigger_time:
            self.balanceNll = 0

    def forward(
        self,
        input_ids,
        input_ids_enc,
        input_ids_ae_dec,
        attention_mask,
        attention_mask_enc,
        attention_mask_ae_dec,
        labels,
        labels_ae,
        **kwargs
    ):

        bs = input_ids.size(0)

        if self.beta > 0 and self.alpha == 0:
            # pretrain ae

            ae_outs = self.aemodel(
                input_ids_enc=input_ids_enc,
                attention_mask_enc=attention_mask_enc,
                input_ids_dec=input_ids_ae_dec,
                attention_mask_dec=attention_mask_ae_dec,
                labels=labels_ae
            )
            recloss = ae_outs.loss

            return CausalLMOutput(
                loss=recloss)


        main_dec_outs = self.main_decoder(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
        nllloss = main_dec_outs.loss

        if self.alpha > 0:
            main_dec_lhs = main_dec_outs.hidden_states[-1]  # bs seqlen h
            # 'CausalLMOutputWithCrossAttentions' object has no attribute 'last_hidden_state'

            is_ztokens = self.z_start_id <= input_ids
            #  < self.z_end_id
            
            main_hidden_z = main_dec_lhs[is_ztokens]
            main_hidden_z = self.proj(main_hidden_z)

            if self.beta == 0:
                ae_outs = self.aemodel(
                input_ids_enc=input_ids_enc,
                attention_mask_enc=attention_mask_enc,
                input_ids_dec=input_ids_ae_dec,
                attention_mask_dec=attention_mask_ae_dec,
                labels=None
                )
                recloss = torch.zeros_like(nllloss)
            else:
                ae_outs = self.aemodel(
                input_ids_enc=input_ids_enc,
                attention_mask_enc=attention_mask_enc,
                input_ids_dec=input_ids_ae_dec,
                attention_mask_dec=attention_mask_ae_dec,
                labels=labels_ae
                )
                recloss = ae_outs.loss

            # print(main_hidden_z.shape, ae_outs.hidden_z.shape)
            target_z = ae_outs.hidden_z.reshape(-1, ae_outs.hidden_z.size(-1)).detach()

            if self.znorm:
                rep_std = torch.std(main_hidden_z, dim=-1, keepdim=True)
                rep_mean = torch.mean(main_hidden_z, dim=-1, keepdim=True)
                main_hidden_z = (main_hidden_z - rep_mean) / rep_std

                rep_std = torch.std(target_z, dim=-1, keepdim=True)
                rep_mean = torch.mean(target_z, dim=-1, keepdim=True)
                target_z = (target_z - rep_mean) / rep_std

            mseloss = self.mseloss(main_hidden_z, target_z)

            tloss = self.alpha * mseloss * self.balanceMse \
                + self.beta * recloss * self.balanceRec \
                + nllloss * self.balanceNll

            print(f"{self.training}; mseloss = {self.alpha} * {mseloss:.6f} * {self.balanceMse:.6f}, recloss = {self.beta} * {recloss:.6f} * {self.balanceRec:.6f}, nllloss = {nllloss:.6f} * {self.balanceNll:.6f}")
        else:
            tloss = nllloss

        return CausalLMOutput(
            loss=tloss if self.training else nllloss,
            logits=main_dec_outs.logits,
        )
