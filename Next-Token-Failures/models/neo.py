import math
import os
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

from transformers.models.gpt_neo import GPTNeoForCausalLM, GPTNeoModel
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoAttention
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.modeling_outputs import (
    CausalLMOutput,
)
@dataclass
class AEOutput(ModelOutput):
 
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_z : torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    
from transformers import (
    AutoModel,
    PreTrainedModel, 
    GPT2PreTrainedModel, 
    GPTNeoPreTrainedModel,
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

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

logger = logging.get_logger(__name__)


class LnPrefixEncoder(nn.Module):
    def __init__(self, config, model_args):
        super().__init__()
        # Config
        # self.prefix_seq_len = config.prefix_seq_len
        self.input_dim = model_args.zdim
        self.hidden_dim = config.hidden_size

        # self.prefix_dropout_rate = getattr(config, "prefix_dropout_rate", 0.0)
        self.prefix_dropout_rate = 0.1
        self.prefix_seq_len = model_args.ztokens

        # Model
        # self.input_tokens = torch.arange(self.prefix_seq_len).long()
        # self.prefix_wte = nn.Embedding(self.prefix_seq_len, config.input_dim)

        # Since prefix-tuning append prefix to each layer, the shape is prefix_seq_len, n_layer, 2(query,key), hidden_size
        '''
        self.prefix_mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, config.n_layer * 2 * config.hidden_size),
        )
        '''
        self.match_n_layer = model_args.shallow_decoder_n_layer
        self.prefix_mlp = nn.Linear(
            self.input_dim, self.match_n_layer * 2 * config.hidden_size)

        self.prefix_dropout = nn.Dropout(self.prefix_dropout_rate)

        self.lns = nn.ModuleList(
            [
                nn.LayerNorm(
                    2 * config.hidden_size,
                    eps = config.layer_norm_epsilon
                ) 
                for _ in range(self.match_n_layer)
            ]
        )

        
        self.match_n_head = config.num_heads
        self.match_n_embd = config.hidden_size // config.num_heads
    def forward(
        self,
        input_embd,
    ):
        """
        Return query & key values from prefix
        """
        # input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        # Forward
        # input_embd = self.prefix_wte(input_tokens)

        # B NZ H
        batch_size = input_embd.size(0)

        past_key_values = self.prefix_mlp(input_embd)

        past_key_values = past_key_values.view(batch_size, self.prefix_seq_len, self.match_n_layer, -1)

        past_key_values_new = []
        for il, ln in enumerate(self.lns):
            past_key_values_new.append(ln(past_key_values[:, :, il]))
        past_key_values = torch.stack(past_key_values_new, dim=2)

        # Resize
        past_key_values = past_key_values.view(
            batch_size,
            self.prefix_seq_len,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        # Dropout
        # past_key_values = self.prefix_dropout(past_key_values)

        # Transpose -> [match_n_layer*2, batch_size, match_n_head, prefix_seq_len, match_n_embd]
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4])
        past_key_values = torch.split(past_key_values, 2)
        
        all_kvs = ()
        for i in range(len(past_key_values)):
            kvpair = (past_key_values[i][0], past_key_values[i][1])
            all_kvs += (kvpair,)

        return all_kvs
    
class PrefixEncoder(nn.Module):
    def __init__(self, config, model_args):
        super().__init__()

        self.input_dim = model_args.zdim
        self.hidden_dim = config.hidden_size

        self.prefix_seq_len = model_args.ztokens
        self.match_n_layer = model_args.shallow_decoder_n_layer
        
        self.prefix_mlp = nn.Linear(
            self.input_dim, self.match_n_layer * 2 * config.hidden_size)


        self.match_n_head = config.num_heads
        self.match_n_embd = config.hidden_size // config.num_heads

    def forward(
        self,
        input_embd
    ):
        
        batch_size = input_embd.size(0)

        past_key_values = self.prefix_mlp(input_embd)

        past_key_values = past_key_values.view(batch_size, self.prefix_seq_len, self.match_n_layer, -1)



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
        #print("finally:")
        #print(all_kvs[0][0])
        #print(f"len all_kvs: {type(all_kvs), len(all_kvs)}")
        #print(f"len all_kvs[0]: {type(all_kvs[0]),len(all_kvs[0])}")
        #print(f"shape of k/v : {type(all_kvs[0][0]), all_kvs[0][0].shape, all_kvs[0][1].shape}")
        #assert 0

        return all_kvs


class AEDecoder(GPTNeoModel):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        if inputs_embeds is None:
            #added
            with torch.no_grad():
                inputs_embeds = self.wte(input_ids)
        #added
        with torch.no_grad():
            position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Attention mask.
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_length)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        
        
class AE(GPTNeoForCausalLM):
    def __init__(self, config, model_args):
        super().__init__(config)
        self.ztokens = model_args.ztokens
        
        self.shallow_decoder_config = deepcopy(self.config)
        self.shallow_decoder_config.num_layers = model_args.shallow_decoder_n_layer
        
        self.decoder = AEDecoder(self.shallow_decoder_config)
        
        self.cross_attention = GPTNeoAttention(config)
        
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        self.prefix_encoder = LnPrefixEncoder(config, model_args)
        self.proj = nn.Linear(config.hidden_size, model_args.zdim, bias=False)
        self.post_init()
        
    def build_ed(self, model_args, main_decoder):
        if model_args.from_scratch:
            self.decoder = AEDecoder(self.shallow_decoder_config)
        else:
            self.decoder = AEDecoder.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=self.shallow_decoder_config
            )
        print(self.shallow_decoder_config)

        
        new_vocab_size = main_decoder.transformer.wte.weight.size(0)
        
        self.decoder.resize_token_embeddings(new_vocab_size)
        # self.transformer.resize_token_embeddings(new_vocab_size)
               
        self.decoder.wte = main_decoder.transformer.wte
        self.decoder.wpe = main_decoder.transformer.wpe
        self.lm_head = main_decoder.lm_head
        
        self.transformer = main_decoder.transformer

        self.zwte = nn.Embedding(self.transformer.config.vocab_size, self.transformer.embed_dim)
        self.zwpe = nn.Embedding(self.transformer.config.max_position_embeddings, self.transformer.embed_dim)
        self.zwte.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.zwpe.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        
    def forward(
        self,
        input_ids_enc,
        input_ids_enc_z,
        labels_enc,
    ):

        input_shape_z = input_ids_enc_z.size()
        input_ids_enc_z = input_ids_enc_z.view(-1, input_shape_z[-1])
        position_ids_z = torch.arange(0, input_shape_z[-1], dtype=torch.long, device=input_ids_enc_z.device)
        position_ids_z = position_ids_z.unsqueeze(0).view(-1, input_shape_z[-1])
        inputs_embeds_z = self.zwte(input_ids_enc_z)
        position_embeds_z = self.zwpe(position_ids_z)
        hidden_states_z = inputs_embeds_z + position_embeds_z
        

        with torch.no_grad():

            enc_outs = self.transformer(
                input_ids = input_ids_enc
            )

    
        
        enc_lhs = enc_outs.last_hidden_state 
        residual = hidden_states_z
        hidden_states_z = self.ln_1(hidden_states_z)        
        
        
        match_n_embd = self.config.hidden_size // self.config.num_heads
        pkv = enc_lhs.view(
            enc_lhs.size(0), # bs
            enc_lhs.size(-2), # seqlen
            self.config.num_heads,
            match_n_embd,
        )

        pkv = pkv.permute([0, 2, 1, 3])
        pkv = (pkv, pkv.clone(),)
        
        cross_outs = self.cross_attention(
            hidden_states = hidden_states_z,
            layer_past = pkv
        )
        
        hidden_z = cross_outs[0] + residual
        # hidden_z = self.ln_2(hidden_z)
        hidden_z = self.proj(hidden_z)
        
        past_key_values = self.prefix_encoder(hidden_z)
        

        dec_outs = self.decoder(
            input_ids = input_ids_enc,
            past_key_values = past_key_values,
            output_hidden_states = True,
            output_attentions = False
        )

        lhs = dec_outs.last_hidden_state
        
        lm_logits = self.lm_head(lhs)
            
        loss = None
        if labels_enc is not None:
            # move labels to correct device to enable model parallelism
            labels_enc = labels_enc.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels_enc[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        rec_loss = loss
        return AEOutput(
            loss = rec_loss,
            hidden_z = hidden_z
        )

       
        
    
class NewTModel(GPTNeoPreTrainedModel):
    def __init__(self, config, model_args):
        print("*******************************")
        print(config)
        if model_args.zdim == -1:
            model_args.zdim = config.hidden_size
        print(model_args)
        
        super().__init__(config)

        self.ztokens = model_args.ztokens
        
        self.alpha = model_args.alpha
        self.beta = model_args.beta
        
        self.model_args = model_args
        
        self.mseloss = F.smooth_l1_loss
        #self.mseloss = nn.MSELoss()
            
        self.main_decoder = GPTNeoForCausalLM(self.config)
        self.aemodel = AE(deepcopy(config), model_args)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.proj  = nn.Linear(config.hidden_size, model_args.zdim, bias=False)
        
        
    def build_ed(self, len_tokenizer):
        print("trying to build ed...")
        if not self.model_args.from_scratch:
            print("using pretrained model...")
            self.main_decoder = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
                config=self.config
            )
        else:
            print("using model from scratch...")
        self.resize_token_embeddings(len_tokenizer)
        self.aemodel.build_ed(self.model_args, self.main_decoder)

        
    def resize_token_embeddings(self, len_t):
        self.main_decoder.resize_token_embeddings(len_t)
        
    def generate(self, *args, **kwargs):
        return self.main_decoder.generate(*args, **kwargs)
    
            
    def forward(
        self,
        input_ids, 
        labels, 
        input_ids_enc,
        labels_enc,
        input_ids_enc_z,
        pos_mask, 
        **kwargs
    ):

        bs = input_ids.size(0)

        main_dec_outs = self.main_decoder(
            input_ids = input_ids, 
            labels = labels,
            output_hidden_states = True,
            output_attentions = True
        )


        main_dec_lhs = main_dec_outs.hidden_states[-1] # bs seqlen h
        #'CausalLMOutputWithCrossAttentions' object has no attribute 'last_hidden_state' 

        main_hidden_z = main_dec_lhs[pos_mask]
        main_hidden_z = self.proj(main_hidden_z)
        
        ae_outs = self.aemodel(
            input_ids_enc = input_ids_enc,
            input_ids_enc_z = input_ids_enc_z,
            labels_enc = labels_enc
        )

        #print(main_hidden_z.shape, ae_outs.hidden_z.shape)
        mseloss = self.mseloss(main_hidden_z.float(), ae_outs.hidden_z.reshape(-1, ae_outs.hidden_z.size(-1)).detach().float())

        recloss = ae_outs.loss
        nllloss = main_dec_outs.loss
        

        tloss = self.alpha * mseloss  \
            + self.beta * recloss  \
            + nllloss 
            
        with open(f'./balance_logs/{self.model_args.spname}.txt', 'a') as f:
           f.write(f"{self.training}; mseloss = {self.alpha} * {mseloss:.6f}, recloss = {self.beta} * {recloss:.6f}, nllloss = {nllloss:.6f}\n")

        preds = main_dec_outs.logits.argmax(dim=-1)
        
        preds = preds[:, :-1]
        labels = labels[:,1:]
        #print(f"preds: {preds[labels != -100]}")
        #print(f"golds: {labels[labels != -100]}")
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
        
