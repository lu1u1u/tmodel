from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers import (
    AutoModel,
    PreTrainedModel,
    GPT2PreTrainedModel,
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

from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block
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


from peft import get_peft_model


class AE(PreTrainedModel):
    def __init__(self, config, lora_config, model_args):
        super().__init__(config)
        self.ztokens = config.ztokens
        
        self.zwte = nn.Embedding(self.ztokens, config.n_embd)
        
        if model_args.from_scratch:
            self.encoder = AutoModelForCausalLM.from_config(config)
            self.decoder = AutoModelForCausalLM.from_config(config)
        else:  
            self.encoder = AutoModelForCausalLM.from_pretrained(
                model_args.ae_model_name_or_path,
                config=config,
                #use_flash_attention_2=True
            )
            self.decoder = AutoModelForCausalLM.from_pretrained(
                model_args.ae_model_name_or_path,
                config=config,
                #use_flash_attention_2=True
        )
            
        self.encoder.resize_token_embeddings(config.len_tokenizer)
        self.decoder.resize_token_embeddings(config.len_tokenizer)
        if lora_config:
            self.encoder = get_peft_model(self.encoder, lora_config)

        self.proj = nn.Linear(config.hidden_size, config.zdim, bias=False)
        self.up = nn.Linear(config.zdim, config.hidden_size, bias=False)

        self.z_start_id = config.z_start_id
        self.z_end_id = self.z_start_id + self.ztokens

    def forward(
        self,
        input_ids_enc,
        input_ids_dec,
        labels
    ):
        bz = input_ids_enc.size(0)

        z_idx = torch.arange(0, self.ztokens, dtype=torch.long, device=input_ids_enc.device)

        # position_ids_z = z_idx.unsqueeze(0).repeat(bz, 1)
        input_ids_enc_z = z_idx.unsqueeze(0).repeat(bz, 1)
        inputs_embeds_z = self.zwte(input_ids_enc_z)

        # position_embeds_z = self.zwpe(position_ids_z)
        # hidden_states_z = inputs_embeds_z + position_embeds_z

        input_ids_embeds = self.encoder.transformer.wte(input_ids_enc)
        
        # replace trainable z embeds
        is_ztokens = self.z_start_id <= input_ids_enc

        input_ids_embeds[is_ztokens] = inputs_embeds_z.view(-1, inputs_embeds_z.size(-1))

        enc_outs = self.encoder.transformer(
            input_ids=None,
            inputs_embeds=input_ids_embeds,
        )
        hidden = enc_outs[0]

        hidden_z = hidden[is_ztokens].view(bz, -1, hidden.size(-1))
        hidden_down = self.proj(hidden_z)

        # add noise?

        loss = None
        if labels is not None:

            hidden_up = self.up(hidden_down)
            # without z tokens
            input_ids_embeds = self.decoder.transformer.wte(input_ids_dec)
            
            # print(hidden_up.shape, input_ids_embeds.shape, hidden_up.device, input_ids_embeds.device)
            # exit()
            input_ids_embeds_with_z = torch.cat((hidden_up, input_ids_embeds), dim=1)
            # B Z
            
           


            self.decoder.eval()
            dec_outs = self.decoder.transformer(
                inputs_embeds=input_ids_embeds_with_z,
                # input_ids=input_ids_enc,
                # past_key_values=past_key_values,
                output_hidden_states=True,
                output_attentions=False
            )

            lhs = dec_outs.last_hidden_state

            lm_logits = self.decoder.lm_head(lhs)

            lm_logits = lm_logits[:, hidden_up.size(1):]
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
        
        if self.beta > 0:
            self.aemodel = AE(deepcopy(config), lora_config, model_args)
            
        if model_args.from_scratch:
            self.main_decoder = AutoModelForCausalLM.from_config(config)
        else:
            self.main_decoder = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                config=config,
                #use_flash_attention_2=True
            )
        self.main_decoder.resize_token_embeddings(config.len_tokenizer)
        self.proj = nn.Linear(config.hidden_size, config.zdim, bias=False)

        

        self.softmax = nn.Softmax(dim=-1)


    def freeze(self):
        if hasattr(self, "aemodel"):
            self.aemodel.decoder.requires_grad_(False)


    def resize_token_embeddings(self, len_t):
        self.main_decoder.resize_token_embeddings(len_t)

    def generate(self, *args, **kwargs):
        return self.main_decoder.generate(*args, **kwargs)

   

    def forward(
        self,
        input_ids,
        input_ids_enc,
        input_ids_ae_dec,
        labels,
        labels_ae,
        **kwargs
    ):



        if self.beta > 0 and self.alpha == 0:
            # pretrain ae
            assert 0, 'pretraining not impl. in pitfall'
            ae_outs = self.aemodel(
                input_ids_enc=input_ids_enc,
                input_ids_dec=input_ids_ae_dec,
                labels=labels_ae
            )
            recloss = ae_outs.loss

            return CausalLMOutput(
                loss=recloss)


        main_dec_outs = self.main_decoder(
            input_ids=input_ids,
            labels=labels,
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
                input_ids_dec=input_ids_ae_dec,
                labels=None
                )
                recloss = torch.zeros_like(nllloss)
            else:
                ae_outs = self.aemodel(
                input_ids_enc=input_ids_enc,
                input_ids_dec=input_ids_ae_dec,
                labels=labels_ae
                )
                recloss = ae_outs.loss

            target_z = ae_outs.hidden_z.reshape(-1, ae_outs.hidden_z.size(-1)).detach()

            if self.znorm:
                rep_std = torch.std(main_hidden_z, dim=-1, keepdim=True)
                rep_mean = torch.mean(main_hidden_z, dim=-1, keepdim=True)
                main_hidden_z = (main_hidden_z - rep_mean) / rep_std

                rep_std = torch.std(target_z, dim=-1, keepdim=True)
                rep_mean = torch.mean(target_z, dim=-1, keepdim=True)
                target_z = (target_z - rep_mean) / rep_std

            mseloss = self.mseloss(main_hidden_z, target_z)

            tloss = self.alpha * mseloss \
                + self.beta * recloss \
                + nllloss
        else:
            tloss = nllloss
            
        preds = main_dec_outs.logits.argmax(dim=-1)
        
        preds = preds[:, :-1]
        labels = labels[:,1:]
        acc = self.accuracy(preds, labels)
        return main_dec_outs.logits, tloss if self.training else nllloss, acc
    
    
    def accuracy(self, preds, labels):
        bz = labels.size(0)
        #print(labels)
        labels = labels[labels!=-100].reshape(bz, -1)
        #print(labels)
        preds = preds[:, -labels.size(1):]
        #print(preds)
        #print(preds.shape, labels.shape)

        correct = preds.eq(labels).to(torch.float)
        seq_correct = torch.sum(correct, dim=1).eq(labels.size(1)).float()
        acc = torch.mean(seq_correct)
        per_token_acc = correct.mean(dim=0)
        return {
            'acc' : acc,
            'token_acc' : per_token_acc
        }
