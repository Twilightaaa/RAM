import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
 
from dataclasses import dataclass, field
from typing import Optional, List
from peft import (
    get_peft_model,
)

from safetensors.torch import load_file
from icecream import ic as pprint
from math import sqrt
import random
from typing import List, Optional, Tuple, Dict, Any
import json
import torch.distributed as dist

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None, metadata={"help": "Path to model."})
    target_model: str = field(default=None, metadata={"help": "Path to target model."})
    
    lora_r: int = 128
    lora_dropout: float = 0.05
    lora_alpha: int = 32

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    is_train: bool = True
    ppl_memory: bool = True
    mrMR: bool = True
    reconstruct: bool = True
    alpha: float = 1.0
    compressor_hidden_size: int = 4096
    target_llm_hidden_size: int = 4096
    mem_size: int = 32
    segment_size: int = 200
    benchmark_metric: str = None
    compressor_hidden_layers: int = 2
    merge_size: int = 4
    head_num: int = 8
    scale: float = 0.5
    mean: bool = True
    num_mem_fusion_layers: int = 1
    mem_lora: bool = True

    segment_size: int = 200
    benchmark_metric: str = "accuracy"
    post_append: bool = False
    is_random: bool = True
    
    split: bool = False
    autoregressive: bool = True
    
    early_stopping_patience: int = 1
    fine_tune: bool = False
    
    lm_ratio: float = 0.5
    leave_len: int = 100
    
    prefix_type: str = 'rs_prefix'
    full: bool = False
    
    keft: bool = False
    restatement_ratio: float = 0.5
    
    icae_infer: bool = False
    
    use_transform_layer: bool = True

    # 启用梯度检查点
    gradient_enable: bool = True

    # add for tkdr
    launch_tkdr: bool = True
    key_percentage: float = 0.005
    merge_sizes: str = "8"
    adaptive_pick: bool = True
    tau: float = 0.85

    lamda_select: bool = True
    lamda_merge: bool = True

    empty: bool = True

    lora_the_encoder: bool = False

    # add for ablation study
    coarse_grained_on: bool = True
    fine_grained_on: bool = True
    redun_coarse: bool = True
    redun_fine: bool = True

    # draw the tsne figure
    draw: bool = True
    fig_name: str = "test.png"

    # add for paralizable reasoning
    index: int = 0

    # add for encore
    enable_encore: bool = True
    top_p: float = 0.9
    uniform_distribution: bool = True
    top_p_list: str = "0.9"
    use_contrastive_loss: bool = True

    use_only_org_tokens: bool = False
    use_mean_compressed_tokens: bool = False
    use_all_compress: bool = False

@dataclass
class RAMArguments:
    compressor_path: str = None
    lm_model_path: str = None
    lm_model_name: str = 'longchat'
    num_compressor_layers: int = 4
    num_compressor_encoder_layers: int = 2
    fix_compressor_mlp_parameters: bool = False
    num_attention_heads: int = 32
    attn_doc_topp: float = 0.25
    # compressor_hidden_size: int = 4096
    # lm_model_hidden_size: int = 5120
    generation_split_token: str = None
    pool_window_size: int = 4
    random_pool_window_size: bool = False
    cand_pool_window_sizes: List[int] = None

    label_pad_token_id: int = -100

    # inference args
    pw_window_sizes: List[int] = None
    data_path: str = None
    train_data_path: str = None
    num_eval_documents: int = 5

    num_gold_documents: int = 1
    use_answer_as_target: bool = False
    instruction_name: str = 'base'
    # instruction_name: str = 'summary'
    gold_first_for_kd: bool = False

    min_num_documents: int = 1
    max_num_documents: int = 5
    random_num_documents: bool = False

    max_new_tokens: int = 100
    max_doc_tokens: int = 512
    
    restatement: bool = True

def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False
        
def freeze_mlp(model):
    for name, param in model.named_parameters():
        if 'mlp' in name:
            param.requires_grad = False

def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable: {100 * trainable_parameters / all_param} %")

def write_txt(name, item):
    with open(name, "a+") as f:
        f.write(str(item))
        f.write("\n")
    
def select_key_values(key_values, need_idx):
    return [
        [
            k[..., need_idx, :],
            v[..., need_idx, :],
        ]
        for k, v in key_values
    ]

def get_embeding_from_block(
    text,
    model,
    tokenizer
):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**encoded_input)
    
    last_hidden_states = output.last_hidden_state
    # 忽略特殊标记[CLS]和[SEP]，只对实际的文本标记进行平均
    mean_pooled_embedding = torch.mean(last_hidden_states[:, 1:-1, :], dim=1)[0][0]
    return mean_pooled_embedding

def weight_to_mem_size(
    weights,
    mem_size      
):
    allocated_mem_sizes = []
    for idx, w in enumerate(weights):
        if idx == (len(weights) - 1):
            s = sum(allocated_mem_sizes)
            allocated_mem_sizes.append(mem_size - s)
        else:
            allocated_mem_sizes.append(int(w * mem_size))

    return allocated_mem_sizes

def dynamic_allocator(
    segments, 
    query,
    mem_size,
    model,
    tokenizer
):
    query_embed = get_embeding_from_block(
        query,
        model,
        tokenizer
    )
    similaritys = []
    sum_similarity = 0
    for segment in segments:
        segment_embed = get_embeding_from_block(
            segment,
            model,
            tokenizer
        )

        similarity = query_embed @ segment_embed
        sum_similarity += similarity
        similaritys.append(similarity)

    normalize_similaritys = [1.0 * e / sum_similarity for e in similaritys]
    allocated_mem_sizes = weight_to_mem_size(normalize_similaritys, mem_size)
    return allocated_mem_sizes

class MemoryFusion(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MemoryFusion, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换矩阵
        self.q_proj = nn.Linear(d_model, d_model).to(torch.bfloat16)
        self.k_proj = nn.Linear(d_model, d_model).to(torch.bfloat16)
        self.v_proj = nn.Linear(d_model, d_model).to(torch.bfloat16)
        self.fct = nn.Linear(d_model, d_model).to(torch.bfloat16)
        
    def forward(self, X, attention_mask=None):
        # pprint(X.dtype)
        X = X.to(torch.bfloat16)
        # exit(0)
        batch_size, seq_len, d_model = X.size()
        
        # 线性变换得到 Q, K, V
        Q = self.q_proj(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.bfloat16))
        
        # 应用 attention_mask
        if attention_mask is not None:
            # 扩展 attention_mask 的维度以匹配 scores 的形状
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # 形状为 [batch_size, 1, 1, seq_len]
            # 将需要忽略的位置的分数设为一个很小的值
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # 应用 Softmax 得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attention_weights, V)
        
        # 拼接多头输出
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 最终线性变换
        # output.shape: [batch_size, seq_len, d_model]
        output = self.fct(output)
        
        # pooling
        final_output = torch.mean(output, dim=1)
        
        return final_output

class LLMCompressor(nn.Module):
    def __init__(
            self,
            model_args, 
            training_args,
            lora_config,
            max_doc_tokens
        ):
        
        super().__init__()

        self.model_args = model_args
        self.training_args = training_args
        
        self.model_name = model_args.model_name_or_path
        self.target_llm = model_args.target_model

        self.training = training_args.is_train
        self.empty = training_args.empty
        self.ppl_memory = training_args.ppl_memory
        self.post_append = training_args.post_append
        # pprint(self.ppl_memory)
        self.mrMR = training_args.mrMR
        self.alpha = training_args.alpha
        self.reconstruct = training_args.reconstruct
        self.merge_size = training_args.merge_size
        self.compressor_hidden_size = training_args.compressor_hidden_size
        self.llm_hidden_size = training_args.target_llm_hidden_size
        self.mem_size = training_args.mem_size
        self.segment_size = training_args.segment_size
        self.mean = training_args.mean
        self.split = training_args.split
        self.autoregressive = training_args.autoregressive
        self.fine_tune = training_args.fine_tune
        self.lm_ratio = training_args.lm_ratio
        self.scale = training_args.scale
        self.leave_len = training_args.leave_len
        self.full = training_args.full
        self.keft = training_args.keft
        self.restatement_ratio = training_args.restatement_ratio
        self.use_transform_layer = training_args.use_transform_layer
        self.enable_encore = training_args.enable_encore
        self.top_p = training_args.top_p
        self.uniform_distribution = training_args.uniform_distribution
        self.adaptive_temperature = nn.Parameter(torch.zeros(1))
        self.use_contrastive_loss = training_args.use_contrastive_loss
        self.use_only_org_tokens = training_args.use_only_org_tokens
        self.use_mean_compressed_tokens = training_args.use_mean_compressed_tokens
        self.use_all_compress = training_args.use_all_compress

        if training_args.lamda_select:
            self.alpha_select = nn.Parameter(torch.tensor(1.0))
            self.beta_select = nn.Parameter(torch.tensor(1.0))

        if training_args.lamda_merge:
            self.alpha_merge = nn.Parameter(torch.tensor(1.0))
            self.beta_merge = nn.Parameter(torch.tensor(1.0))

        self.compressor_config = AutoConfig.from_pretrained(self.model_name)
        self.fusion_layer_config = AutoConfig.from_pretrained(self.target_llm)
        self.decoder_config = AutoConfig.from_pretrained(self.target_llm)
        
        self.compressor_size = self.compressor_config.hidden_size
        self.target_size = self.fusion_layer_config.hidden_size
        
        # NOTE: we delete the operations for modify llm encoder
        self.compressor_config.num_hidden_layers = training_args.compressor_hidden_layers
        # NOTE: we should scaling llama rope
        # orig_ctx_len = getattr(self.compressor_config, "max_position_embeddings")
        # if max_doc_tokens > orig_ctx_len:
        #     scaling_factor = float(math.ceil(max_doc_tokens / orig_ctx_len))
        #     self.compressor_config.rope_scaling = {"type": "linear", "factor": scaling_factor}

        self.gradient_checkpointing_enable = training_args.gradient_enable
        self.merge_sizes = training_args.merge_sizes
        self.lora_the_encoder = training_args.lora_the_encoder

        # add for ablation study
        self.coarse_grained_on = training_args.coarse_grained_on
        self.fine_grained_on = training_args.fine_grained_on
        self.redun_coarse = training_args.redun_coarse
        self.redun_fine = training_args.redun_fine

        # add for tkdr
        self.launch_tkdr = training_args.launch_tkdr
        
        # NOTE: we delete the operations for modify llm encoder
        self.llm_encoder = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=self.compressor_config
        )
        
        self.llm_encoder = self.llm_encoder.to(torch.bfloat16)

        if self.lora_the_encoder:
            self.llm_encoder = get_peft_model(self.llm_encoder, lora_config)

        self.is_random = training_args.is_random
        if self.post_append:
            self.semantic_alignment_layer = nn.Linear(self.compressor_size \
                                                , self.target_size).to(dtype=torch.float16)
            
        # NOTE: we always load the semantic alignment layer
        self.semantic_alignment_layer = nn.Linear(self.compressor_size \
                                    , self.target_size).to(dtype=torch.float16)
        
        # load memory fusion layer
        if not self.post_append:
            self.fusion_layer_config.num_hidden_layers = training_args.num_mem_fusion_layers
            if self.use_transform_layer:
                if self.compressor_size != self.target_size:
                    self.dimension_alignment_layer = nn.Linear(self.compressor_size \
                                                        , self.target_size).to(dtype=torch.bfloat16) 
                     
                self.memory_fusion_layer = AutoModelForCausalLM.from_pretrained(
                    self.target_llm,
                    config=self.fusion_layer_config
                )
                self.memory_fusion_layer = self.memory_fusion_layer.to(torch.bfloat16)
                if training_args.mem_lora:
                    self.memory_fusion_layer = get_peft_model(self.memory_fusion_layer, lora_config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.vocab_size = self.llm_encoder.config.vocab_size   

        if self.post_append:
            self.vocab_size_with_mem = self.vocab_size + self.mem_size
            self.llm_encoder.resize_token_embeddings(self.vocab_size + self.mem_size)
            self.memory_token_embed = nn.Embedding(self.mem_size, self.compressor_hidden_size, padding_idx=None)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.ppl_fct = nn.CrossEntropyLoss(reduction="none")
        self.icae_infer = training_args.icae_infer

        if self.post_append:
            self.memory_sequence = torch.arange(self.vocab_size, self.vocab_size + self.mem_size)
            
        if  self.training:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.target_llm)
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.target_llm,
                config=self.decoder_config,
                torch_dtype=torch.bfloat16
            )

            self.llm_vocab_size = self.llm.config.vocab_size
            
            self.init()
        
        if self.gradient_checkpointing_enable:

            if hasattr(self, 'memory_fusion_layer'):
                self.memory_fusion_layer.use_cache = False
                self.memory_fusion_layer.config.use_cache = False
                self.memory_fusion_layer.enable_input_require_grads()
                self.memory_fusion_layer.gradient_checkpointing_enable()

            if hasattr(self, 'llm'):
                self.llm.use_cache = False
                self.llm.config.use_cache = False
                self.llm.enable_input_require_grads()
                self.llm.gradient_checkpointing_enable()

    def generate_merge_size(
        self
    ):
        if "," not in self.merge_sizes:
            numbers = [int(self.merge_sizes)]
        else:
            numbers = [int(e) for e in self.merge_sizes.split(",")]

        sampled_number = random.choice(numbers)
        return sampled_number
    
    def generate_post_append_size(
        self  
    ):
        numbers = [64, 128]

        sampled_number = random.choice(numbers)
        return sampled_number

    def infoNCE_loss(
        self, 
        query_embed, 
        memory_embed, 
        other_memorys_embed, 
        temperature=0.1
    ):

        memory_embed = memory_embed.unsqueeze(0)  # [1, 768]
        other_memorys_embed = other_memorys_embed  # [8, 768]

        positive_similarity = torch.matmul(query_embed.unsqueeze(0), memory_embed.T) / temperature  # [1, 1]
        negative_similarity = torch.matmul(query_embed.unsqueeze(0), other_memorys_embed.T) / temperature  # [1, 8]

        all_similarity = torch.cat([positive_similarity, negative_similarity], dim=1)  # [1, 9]

        labels = torch.zeros(all_similarity.size(0), dtype=torch.long).to(query_embed.device)  # [9]

        loss = self.loss_fct(all_similarity, labels)
        return loss

    def freeze_others(
        self
    ):
        for name, param in self.llm.named_parameters():
            if not ('q_proj' in name or 'k_proj' in name or 'v_proj' in name):
            # if not ('q_proj.lora' in name or 'k_proj.lora' in name or 'v_proj.lora' in name):
                param.requires_grad = False

    def init(self):
        self.freeze_others()
        print_trainable_parameters(self)

    def split_segments(
        self,
        context_input_ids,
        memory_sequence,
        input_mask
    ):
        # pprint(context_input_ids.shape)
        batch_size = context_input_ids.shape[0]
        memory_sequence = memory_sequence.repeat(batch_size, 1)
        # pprint(memory_sequence.shape)
        num_segments = math.ceil(context_input_ids.shape[1] * 1.0 / self.segment_size)
        # pprint(self.segment_size)
        # exit(0)
        segments_input_ids = [context_input_ids[:, i * self.segment_size: (i + 1) \
                                        * self.segment_size] for i in range(num_segments - 1)]
        segments_input_mask = [input_mask[:, i * self.segment_size: (i + 1) \
                                        * self.segment_size] for i in range(num_segments - 1)]
        last_segment_start_index = (num_segments - 1) * self.segment_size
        segments_input_ids.append(context_input_ids[:, last_segment_start_index:])
        segments_input_mask.append(input_mask[:, last_segment_start_index:])

        memory_len = math.ceil(memory_sequence.shape[1] * 1.0 / num_segments)
        memory_sequences = [memory_sequence[:, i * memory_len : (i + 1)\
                                            * memory_len] for i in range(num_segments - 1)]
        memory_sequences.append(memory_sequence[:, (num_segments - 1) * memory_len:])
        return segments_input_ids, memory_sequences, segments_input_mask

    def compute_num_segments(self, total_length):
        assert total_length > 0
        num_segments = math.ceil(total_length * 1.0 / self.segment_size)
        return num_segments

    def get_avg_embeds(self, tensor, attention_mask):
        # 1. 扩展 attention_mask 的维度
        attention_mask = attention_mask.unsqueeze(-1)  # [batch_size, merge_size, 1]

        # 2. 应用 attention_mask
        masked_tensor = tensor * attention_mask  # [batch_size, merge_size, hidden_size]

        # 3. 计算有效元素的和
        sum_tensor = masked_tensor.sum(dim=1)  # [batch_size, hidden_size]

        # 4. 计算有效元素的个数
        valid_counts = attention_mask.sum(dim=1)  # [batch_size, 1]

        # 5. 计算平均值
        average_tensor = sum_tensor / valid_counts  # [batch_size, hidden_size]
        return average_tensor

    def generate_pooling_memorys(
        self,
        input_ids,
        input_mask,
        merge_size
    ):
        device = input_ids.device
        if self.is_random:
            merge_size = self.generate_merge_size()
        
        # pprint(input_ids.dtype)
        # pprint(input_ids.shape)
        # # input_ids = input_ids.int()
        # exit(0)
        last_hidden_state = self.llm_encoder(
            input_ids=input_ids,
            attention_mask=input_mask,
            output_hidden_states=True
        ).hidden_states[-1]
        # exit(0)
        
        batch_size = input_ids.shape[0]
        hidden_size = last_hidden_state.shape[2]
        memorys_list = []
        for i in range(batch_size):
            current_mask = input_mask[i]
            is_all_zero = torch.all(current_mask == 0).item()
            # pprint(is_all_zero)
            if is_all_zero:
                sequence_length = last_hidden_state.shape[0]
                if sequence_length % merge_size == 0:
                    pad_length = 0
                else:
                    pad_length = merge_size - (sequence_length % merge_size)
                
                mem_size = int((sequence_length + pad_length) / merge_size)
                memorys = torch.zeros(mem_size, hidden_size).unsqueeze(0).to(device)
                # pprint(memorys.shape)
                memorys_list.append(memorys)
                continue
            
            current_hidden_state = last_hidden_state[i]
            
            select_hidden_state = torch.cat([y.unsqueeze(0) for x, y in zip(current_mask, \
                current_hidden_state) if x == 1], dim=0)
            current_mask = torch.tensor([x for x in current_mask if x == 1]).to(device)
            
            # # select_hidden_state: [seq, hidden_size]
            # generate memorys then transform
            sequence_length = select_hidden_state.shape[0]
            res_len = sequence_length % merge_size
            if res_len != 0:
                res_hidden_state = select_hidden_state[-res_len:,:]
                res_memory = torch.mean(res_hidden_state, dim=0).unsqueeze(0)
                select_hidden_state = select_hidden_state[:-res_len, :]
                
            hidden_state_reshaped = select_hidden_state.reshape(-1, merge_size, hidden_size)
            hidden_state_reshaped = hidden_state_reshaped.to(torch.bfloat16)
            memorys = torch.mean(hidden_state_reshaped, dim=1)
            
            if res_len != 0:
                memorys = torch.cat((memorys, res_memory), dim=0).unsqueeze(0)
            else:
                memorys = memorys.unsqueeze(0)
            
            memorys_list.append(memorys)
            
        # padding the memorys
        m_l = [e.shape[1] for e in memorys_list]
        max_len = max(m_l)
        final_memorys_list = []
        att_mask = torch.ones(batch_size, max_len)
        # pprint(att_mask.shape)
        for idx, e in enumerate(memorys_list):
            # pprint(idx)
            # pprint(e.shape)
            # pad the memorys embeddings
            pad_len = max_len - e.shape[1]
            if pad_len == 0:
                pad_memorys = e
            else:
                pad_embeds = torch.zeros(1, pad_len, e.shape[2]).to(device)
                pad_memorys = torch.cat((e, pad_embeds), dim=1)   
            
            is_all_zero = torch.all(e == 0).item()
            if is_all_zero:
                pad_mask = torch.zeros(max_len).to(device)
                att_mask[idx] = pad_mask
            else:
                if pad_len != 0:
                    pad_mask = torch.zeros(pad_len).to(device)
                    att_mask[idx][-pad_len:] = pad_mask
                    # pprint(pad_len)
                    # pprint(pad_mask.shape)
            final_memorys_list.append(pad_memorys)

        att_mask = att_mask.to(device)
        final_memorys = torch.cat(final_memorys_list, dim=0).to(torch.bfloat16)
        if self.use_transform_layer:
            if self.compressor_size != self.target_size:
                final_memorys = self.dimension_alignment_layer(final_memorys)
                
            aligned_memorys = self.memory_fusion_layer(
                inputs_embeds=final_memorys,
                attention_mask=att_mask,
                output_hidden_states=True
            ).hidden_states[-1]
        else:
            aligned_memorys = final_memorys
        
        return aligned_memorys, att_mask

    def generate_autoregressive_pooling_memorys(
        self,
        input_ids,
        input_mask,
        merge_size,
        pre_mem_embeds=None,
        pre_mem_attention_mask=None
    ):
        # pprint(input_ids.shape)
        # pprint(input_mask.shape)
        device = input_ids.device
        if pre_mem_embeds is not None:
            input_embeds = self.llm.model.embed_tokens(input_ids)
            final_input_embeds = torch.cat((pre_mem_embeds, input_embeds), dim=1)
            final_attention_mask = torch.cat((pre_mem_attention_mask, input_mask), dim=1)
            last_hidden_state = self.llm_encoder(
                inputs_embeds=final_input_embeds,
                attention_mask=final_attention_mask,
                output_hidden_states=True
            ).hidden_states[-1][:, pre_mem_embeds.shape[1]:, :]
        else:
            last_hidden_state = self.llm_encoder(
                input_ids=input_ids,
                attention_mask=input_mask,
                output_hidden_states=True
            ).hidden_states[-1]
        
        batch_size = input_ids.shape[0]
        hidden_size = last_hidden_state.shape[2]
        memorys_list = []
        for i in range(batch_size):
            current_mask = input_mask[i]
            is_all_zero = torch.all(current_mask == 0).item()
            # pprint(is_all_zero)
            if is_all_zero:
                sequence_length = last_hidden_state.shape[0]
                if sequence_length % merge_size == 0:
                    pad_length = 0
                else:
                    pad_length = merge_size - (sequence_length % merge_size)
                
                mem_size = int((sequence_length + pad_length) / merge_size)
                memorys = torch.zeros(mem_size, hidden_size).unsqueeze(0).to(device)
                # pprint(memorys.shape)
                memorys_list.append(memorys)
                continue
            
            current_hidden_state = last_hidden_state[i]
            
            select_hidden_state = torch.cat([y.unsqueeze(0) for x, y in zip(current_mask, \
                current_hidden_state) if x == 1], dim=0)
            
            sequence_length = select_hidden_state.shape[0]
            if sequence_length % merge_size == 0:
                pad_length = 0
            else:
                pad_length = merge_size - (sequence_length % merge_size)
            
            padding = torch.zeros(pad_length, hidden_size).to(device)
            total_hidden_state = torch.cat((select_hidden_state, padding), dim=0)
            # hidden_state_reshaped.shape: [mem_len, merge_size, hidden_size]
            hidden_state_reshaped = total_hidden_state.reshape(-1, merge_size, hidden_size)
            # pprint(hidden_state_reshaped.shape)
            
            hidden_state_reshaped = hidden_state_reshaped.to(torch.bfloat16)
            memorys = self.memory_fusion_layer(
                inputs_embeds=hidden_state_reshaped,
                output_hidden_states=True
            ).hidden_states[-1]
            memorys = torch.mean(memorys, dim=1).unsqueeze(0)
            # pprint(memorys.shape)
            # exit(0)
            memorys_list.append(memorys)
        # exit(0)
            

        # padding the memorys
        m_l = [e.shape[1] for e in memorys_list]
        max_len = max(m_l)
        pprint(max_len)
        final_memorys_list = []
        att_mask = torch.ones(batch_size, max_len)
        # pprint(att_mask.shape)
        for idx, e in enumerate(memorys_list):
            # pprint(idx)
            # pprint(e.shape)
            # pad the memorys embeddings
            pad_len = max_len - e.shape[1]
            if pad_len == 0:
                pad_memorys = e
            else:
                pad_embeds = torch.zeros(1, pad_len, e.shape[2]).to(device)
                pad_memorys = torch.cat((e, pad_embeds), dim=1)   
            
            is_all_zero = torch.all(e == 0).item()
            if is_all_zero:
                pad_mask = torch.zeros(max_len).to(device)
                att_mask[idx] = pad_mask
            else:
                if pad_len != 0:
                    pad_mask = torch.zeros(pad_len).to(device)
                    att_mask[idx][-pad_len:] = pad_mask
                    # pprint(pad_len)
                    # pprint(pad_mask.shape)
            final_memorys_list.append(pad_memorys)

        # pprint(final_memorys_list[0].shape)
        # pprint(final_memorys_list[1].shape)
        # pprint(att_mask)
        # exit(0)
        final_memorys = torch.cat(final_memorys_list, dim=0).to(torch.bfloat16)
        aligned_memorys = final_memorys
        # self.semantic_alignment_layer.weight.data = self.semantic_alignment_layer.weight.data.to(torch.bfloat16)
        # pprint(final_memorys.dtype)
        # pprint(self.semantic_alignment_layer.weight.dtype)
        # exit(0)
        # aligned_memorys = self.semantic_alignment_layer(final_memorys).to(torch.bfloat16)
        att_mask = att_mask.to(device)
        
        if pre_mem_embeds is not None:
            total_memorys = torch.cat((pre_mem_embeds, aligned_memorys), dim=1)
            total_attention_mask = torch.cat((pre_mem_attention_mask, att_mask), dim=1)
        else:
            total_memorys = aligned_memorys
            total_attention_mask = att_mask
        
        return total_memorys, total_attention_mask

    def generate_memorys_then_transform(
        self,
        input_ids,
        input_mask,
    ):
        # pprint(input_ids.shape)
        # pprint(input_mask.shape)
        device = input_ids.device
        last_hidden_state = self.llm_encoder(
            input_ids=input_ids,
            attention_mask=input_mask,
            output_hidden_states=True
        ).hidden_states[-1]
        # pprint(last_hidden_state.dtype)
        
        batch_size = input_ids.shape[0]
        hidden_size = last_hidden_state.shape[2]
        memorys_list = []
        for i in range(batch_size):
            current_mask = input_mask[i]
            is_all_zero = torch.all(current_mask == 0).item()
            # pprint(is_all_zero)
            if is_all_zero:
                sequence_length = last_hidden_state.shape[0]
                if sequence_length % self.merge_size == 0:
                    pad_length = 0
                else:
                    pad_length = self.merge_size - (sequence_length % self.merge_size)
                
                mem_size = int((sequence_length + pad_length) / self.merge_size)
                memorys = torch.zeros(mem_size, hidden_size).unsqueeze(0).to(device)
                # pprint(memorys.shape)
                memorys_list.append(memorys)
                continue
            
            current_hidden_state = last_hidden_state[i]
            
            # pprint(current_hidden_state.shape)
            # pprint(current_mask)
            select_hidden_state = torch.cat([y.unsqueeze(0) for x, y in zip(current_mask, \
                current_hidden_state) if x == 1], dim=0)
            # pprint(select_hidden_state.shape)
            # exit(0)
            
            sequence_length = select_hidden_state.shape[0]
            if sequence_length % self.merge_size == 0:
                pad_length = 0
            else:
                pad_length = self.merge_size - (sequence_length % self.merge_size)
            
            padding = torch.zeros(pad_length, select_hidden_state.shape[1]).to(device)
            total_hidden_state = torch.cat((select_hidden_state, padding), dim=0)
            # hidden_state_reshaped.shape: [mem_len, merge_size, hidden_size]
            hidden_state_reshaped = total_hidden_state.reshape(-1, self.merge_size, select_hidden_state.shape[1])
            # pprint(hidden_state_reshaped.shape)
            
            hidden_state_reshaped = hidden_state_reshaped.to(torch.bfloat16)
            # memorys = self.memory_fusion_layer(
            #     inputs_embeds=hidden_state_reshaped,
            #     output_hidden_states=True
            # ).hidden_states[-1]
            memorys = torch.mean(hidden_state_reshaped, dim=1).unsqueeze(0)
            # pprint(memorys.shape)
            # exit(0)
            memorys_list.append(memorys)
        # exit(0)
            

        # padding the memorys
        m_l = [e.shape[1] for e in memorys_list]
        max_len = max(m_l)
        final_memorys_list = []
        att_mask = torch.ones(batch_size, max_len)
        # pprint(att_mask.shape)
        # for idx, e in enumerate(memorys_list):
        #     pad_len = max_len - e.shape[1]
        #     pad_embeds = torch.zeros(1, pad_len, e.shape[2]).to(device)
        #     pad_memorys = torch.cat((e, pad_embeds), dim=1)

        #     if pad_len != 0:
        #         pad_mask = torch.zeros(pad_len).to(device)
        #         att_mask[idx][-pad_len:] = pad_mask
        #         # pprint(pad_len)
        #         # pprint(pad_mask.shape)
        #     final_memorys_list.append(pad_memorys)
        for idx, e in enumerate(memorys_list):
            # pprint(idx)
            # pprint(e.shape)
            # pad the memorys embeddings
            pad_len = max_len - e.shape[1]
            if pad_len == 0:
                pad_memorys = e
            else:
                pad_embeds = torch.zeros(1, pad_len, e.shape[2]).to(device)
                pad_memorys = torch.cat((e, pad_embeds), dim=1)   
            
            is_all_zero = torch.all(e == 0).item()
            if is_all_zero:
                pad_mask = torch.zeros(max_len).to(device)
                att_mask[idx] = pad_mask
            else:
                if pad_len != 0:
                    pad_mask = torch.zeros(pad_len).to(device)
                    att_mask[idx][-pad_len:] = pad_mask
                    # pprint(pad_len)
                    # pprint(pad_mask.shape)
            final_memorys_list.append(pad_memorys)

        # pprint(final_memorys_list[0].shape)
        # pprint(final_memorys_list[1].shape)
        # pprint(att_mask)
        # exit(0)
        final_memorys = torch.cat(final_memorys_list, dim=0).to(torch.bfloat16)
        # final_memorys.shape: [batch_size, mem_len, hidden_size]
        final_memorys = self.memory_fusion_layer(
                inputs_embeds=final_memorys,
                output_hidden_states=True
            ).hidden_states[-1]
        aligned_memorys = final_memorys
        # self.semantic_alignment_layer.weight.data = self.semantic_alignment_layer.weight.data.to(torch.bfloat16)
        # pprint(final_memorys.dtype)
        # pprint(self.semantic_alignment_layer.weight.dtype)
        # exit(0)
        # aligned_memorys = self.semantic_alignment_layer(final_memorys).to(torch.bfloat16)
        att_mask = att_mask.to(device)
        # pprint(final_memorys.shape)
        # pprint(att_mask.shape)
        # exit(0)
        
        return aligned_memorys, att_mask
    
    def get_mean_query_embeddings(
        self,
        query_hidden_state, # [batch_size, actual_q_len, hidden_size]
        query_mask
    ):
        # 1. 扩展 query_mask 的维度
        mask = query_mask.unsqueeze(-1)  # [batch_size, query_len, 1]
        
        # 2. 对 query_hidden_state 进行掩码操作
        masked_hidden_state = query_hidden_state * mask  # [batch_size, query_len, hidden_size]
        
        # 3. 计算每个样本的有效长度
        valid_length = mask.sum(dim=1)  # [batch_size, 1]
        
        # 4. 对 masked_hidden_state 在 query_len 维度上求和，然后除以有效长度
        mean_query_embeddings = masked_hidden_state.sum(dim=1) / valid_length  # [batch_size, hidden_size]
        return mean_query_embeddings

    def weighted_pooling(
        self,
        mean_query_embedding, 
        hidden_state_reshaped
    ):
        # 1. 计算点积
        # hidden_state_reshaped: [mem_len, merge_size, hidden_size]
        # mean_query_embedding: [hidden_size]
        # 使用 einsum 计算点积
        dot_product = torch.einsum("msh,h->ms", hidden_state_reshaped, mean_query_embedding)  # [mem_len, merge_size]
        scaled_dot_product = dot_product
        
        # # 2. 对点积结果进行缩放，避免 softmax 数值不稳定
        # scale_factor = torch.sqrt(torch.tensor(hidden_state_reshaped.size(-1), dtype=torch.bfloat16))  # sqrt(hidden_size)
        # scaled_dot_product = dot_product / scale_factor  # [mem_len, merge_size]

        # 3. 对缩放后的点积结果在 merge_size 维度上计算 softmax，得到权重
        weights = F.softmax(scaled_dot_product, dim=-1)  # [mem_len, merge_size]

        # 4. 使用权重对 hidden_state_reshaped 进行加权求和
        # weights: [mem_len, merge_size] -> unsqueeze 为 [mem_len, merge_size, 1]
        # hidden_state_reshaped: [mem_len, merge_size, hidden_size]
        weighted_sum = torch.einsum("msh,ms->mh", hidden_state_reshaped, weights)  # [mem_len, hidden_size]

        return weighted_sum
        
    def res_weighted_pooling(
        self,
        mean_query_embedding, 
        res_hidden_state
    ):
        # 1. 计算点积
        # res_hidden_state: [res_len, hidden_size]
        # mean_query_embeddings: [hidden_size]
        # 使用 einsum 计算点积
        dot_product = torch.einsum("lh,h->l", res_hidden_state, mean_query_embedding)  # [res_len]
        scaled_dot_product = dot_product
        
        # # 2. 对点积结果进行缩放，避免 softmax 数值不稳定
        # scale_factor = torch.sqrt(torch.tensor(res_hidden_state.size(-1), dtype=torch.bfloat16))  # sqrt(hidden_size)
        # scaled_dot_product = dot_product / scale_factor  # [res_len]

        # 3. 对缩放后的点积结果计算 softmax，得到权重
        weights = F.softmax(scaled_dot_product, dim=-1)  # [res_len]

        # 4. 使用权重对 res_hidden_state 进行加权求和
        # weights: [res_len] -> unsqueeze 为 [res_len, 1]
        # res_hidden_state: [res_len, hidden_size]
        weighted_sum = torch.einsum("lh,l->h", res_hidden_state, weights)  # [hidden_size]

        return weighted_sum
    
    def sample_ratio(self):
        """
        从 (0, 1] 范围采样一个浮点数（不包括0但包括1），确保所有 DDP 进程采样相同值。
        """
        # Only rank 0 generates the random number
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            if rank == 0:
                ratio = 1.0 - random.random()  # (0, 1]
                ratio_tensor = torch.tensor([ratio], device='cuda')
            else:
                ratio_tensor = torch.tensor([0.0], device='cuda')
            # Broadcast to all ranks
            dist.broadcast(ratio_tensor, src=0)
            return ratio_tensor.item()

        # Single GPU or non-DDP: safe to sample locally
        return 1.0 - random.random()

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(
            self,
            model_output, 
            attention_mask
        ):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def generate_ram_memorys(
        self,
        input_ids_list: List[torch.Tensor],
        input_mask_list: List[torch.Tensor],
        query_ids: torch.Tensor,
        query_input_mask: torch.Tensor,
        accumulation_ratio: float,
        pos_seg_ids: Optional[List[List[int]]] = None,
        compute_contrastive: bool = False,
        remove_padding: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = query_ids.shape[0]
        device = query_ids.device
        dtype = next(self.llm_encoder.parameters()).dtype

        if self.uniform_distribution:
            accumulation_ratio = self.sample_ratio()

        num_segments = len(input_ids_list)

        # ==============================
        # 1. Concatenate query + segments and apply padding
        # ==============================
        unified_ids_list = [query_ids] + input_ids_list
        unified_mask_list = [query_input_mask] + input_mask_list
        total_num = len(unified_ids_list)

        max_len = max(t.shape[1] for t in unified_ids_list)

        def pad_tensor(tensor, target_len, pad_value):
            if tensor.shape[1] < target_len:
                return F.pad(tensor, (0, target_len - tensor.shape[1]),
                            value=pad_value)
            return tensor

        padded_ids = torch.stack([
            pad_tensor(ids, max_len, self.tokenizer.pad_token_id)
            for ids in unified_ids_list
        ], dim=1)

        padded_masks = torch.stack([
            pad_tensor(mask, max_len, 0)
            for mask in unified_mask_list
        ], dim=1)

        all_input_ids = padded_ids.view(batch_size * total_num, max_len)
        all_input_mask = padded_masks.view(batch_size * total_num, max_len)

        # ==============================
        # 2. Encoder forward pass
        # ==============================
        encoder_outputs = self.llm_encoder(
            input_ids=all_input_ids,
            attention_mask=all_input_mask,
            output_hidden_states=True
        )
        all_hidden_states = encoder_outputs.hidden_states[-1]
        all_org_embeds = self.llm_encoder.model.embed_tokens(all_input_ids)

        hidden_size = all_hidden_states.shape[-1]
        all_hidden_states = all_hidden_states.view(batch_size, total_num, max_len, hidden_size)
        all_org_embeds = all_org_embeds.view(batch_size, total_num, max_len, hidden_size)
        all_input_mask = all_input_mask.view(batch_size, total_num, max_len)

        # ==============================
        # 3. Get query representation
        # ==============================
        query_hidden = all_hidden_states[:, 0, :, :]
        query_mask = all_input_mask[:, 0, :]
        query_repr = (query_hidden * query_mask.unsqueeze(-1)).sum(dim=1) / \
                    query_mask.sum(dim=1, keepdim=True).clamp(min=1)

        # ==============================
        # 4. Get segment representation
        # ==============================
        seg_hidden = all_hidden_states[:, 1:, :, :]
        seg_org_embeds = all_org_embeds[:, 1:, :, :]
        seg_mask = all_input_mask[:, 1:, :]

        seg_mask_f = seg_mask.unsqueeze(-1).to(dtype)
        avg_rep_tokens = (seg_hidden * seg_mask_f).sum(dim=2) / \
                        seg_mask.sum(dim=2, keepdim=True).clamp(min=1)

        # ==============================
        # 5. contrastive loss
        # ==============================
        if compute_contrastive and self.training:
            query_norm = F.normalize(query_repr, p=2, dim=-1)
            seg_norm = F.normalize(avg_rep_tokens, p=2, dim=-1)
            sim_matrix = torch.bmm(query_norm.unsqueeze(1), seg_norm.transpose(1, 2)).squeeze(1) / 0.05
            
            pos_mask = torch.zeros(batch_size, num_segments, dtype=torch.bool, device=device)
            if pos_seg_ids is not None:
                for b in range(batch_size):
                    if pos_seg_ids[b]:
                        for idx in pos_seg_ids[b]:
                            if 0 <= idx < num_segments:
                                pos_mask[b, idx] = True

            exp_sim = torch.exp(sim_matrix)
            pos_mask_float = pos_mask.float()
            neg_mask_float = (~pos_mask).float()
            
            pos_sum = (exp_sim * pos_mask_float).sum(dim=1, keepdim=True)
            neg_sum = (exp_sim * neg_mask_float).sum(dim=1, keepdim=True)
            denominator = pos_sum + neg_sum + 1e-8
            
            loss_per_sample = -torch.log((pos_sum + 1e-8) / denominator).squeeze(-1)
            
            valid_mask = pos_mask.any(dim=1)
            num_valid = valid_mask.sum().clamp(min=1)
            
            contrastive_loss = torch.where(valid_mask, loss_per_sample, torch.zeros_like(loss_per_sample)).sum() / num_valid
        else:
            contrastive_loss = 0

        # ==============================
        # 6. Top-k segment selection
        # ==============================
        if self.use_all_compress:
            k = 0
            original_order_keep_mask = torch.zeros(batch_size, num_segments, dtype=torch.bool, device=device)
        else:
            rep_tokens_norm = F.normalize(avg_rep_tokens, p=2, dim=-1)
            query_norm_exp = F.normalize(query_repr.unsqueeze(1), p=2, dim=-1)
            rep_cos_sim = torch.bmm(rep_tokens_norm, query_norm_exp.transpose(1, 2)).squeeze(-1)
            segment_probs = F.softmax(rep_cos_sim, dim=-1)

            k = int(accumulation_ratio * num_segments)

            if k == 0:
                original_order_keep_mask = torch.zeros(batch_size, num_segments, dtype=torch.bool, device=device)
            elif k == num_segments:
                original_order_keep_mask = torch.ones(batch_size, num_segments, dtype=torch.bool, device=device)
            else:
                _, topk_indices = torch.topk(segment_probs, k=k, dim=-1, largest=True, sorted=False)
                original_order_keep_mask = torch.zeros(batch_size, num_segments, dtype=torch.bool, device=device)
                original_order_keep_mask.scatter_(1, topk_indices, True)

        seg_lengths = seg_mask.sum(dim=2)
        max_seg_len = seg_lengths.max().item() if num_segments > 0 else 0
        effective_max_seg_len = max(max_seg_len, 1)
        
        num_kept = original_order_keep_mask.sum().item()
        num_compressed = num_segments * batch_size - num_kept
        avg_seg_len = seg_lengths.float().mean().item()
        total_actual_tokens = seg_mask.sum().item()
        
        # print(f"\n{'='*60}")
        # print(f"[DEBUG] Compression Analysis")
        # print(f"{'='*60}")
        # print(f"Num segments: {num_segments}")
        # print(f"Accumulation ratio: {accumulation_ratio:.2%}")
        # print(f"K (segments to keep): {k}")
        # print(f"Segments kept uncompressed: {num_kept}")
        # print(f"Segments compressed: {num_compressed}")
        # print(f"Average segment length: {avg_seg_len:.1f}")
        # print(f"Max segment length: {max_seg_len}")
        # print(f"Total actual tokens (no padding): {total_actual_tokens}")
        # print(f"Effective max seg len (for padding): {effective_max_seg_len}")
        # print(f"Expected length BEFORE remove_padding: {num_segments * effective_max_seg_len}")
        # print(f"Expected length AFTER remove_padding (theoretical): {int(num_kept * avg_seg_len + num_compressed)}")
        # print(f"{'='*60}\n")

        if self.training:
            compression_rate = 1 - (original_order_keep_mask.sum().item() / original_order_keep_mask.numel())
            mode_str = "ALL_COMPRESS" if self.use_all_compress else "SELECTIVE"
            print(f"[{mode_str}] Compression rate: {compression_rate:.2%}, k={k}/{num_segments}")

        # ==============================
        # 7. Query-guided compression tokens (Skimming)
        # ==============================
        avg_query_norm = F.normalize(query_repr, p=2, dim=-1)
        seg_norm_full = F.normalize(seg_hidden, p=2, dim=-1)
        cos_sim = torch.matmul(seg_norm_full, avg_query_norm.unsqueeze(-1)).squeeze(-1)
        mask_value = torch.finfo(cos_sim.dtype).min
        cos_sim = cos_sim.masked_fill(~seg_mask.bool(), mask_value)
        attn_weights = F.softmax(cos_sim, dim=-1)
        query_guided_compressed_tokens = (attn_weights.unsqueeze(-1) * seg_hidden).sum(dim=2)

        # ==============================
        # 8. Construct two paths (uncompressed / compressed)
        # ==============================
        # uncompressed path: [B, S, L_eff, H]
        all_uncompressed = F.pad(seg_org_embeds, (0, 0, 0, effective_max_seg_len - seg_org_embeds.shape[2]), value=0.0)
        all_uncompressed_masks = F.pad(seg_mask, (0, effective_max_seg_len - seg_mask.shape[2]), value=False)

        # compressed path: [B, S, L_eff, H]
        comp_emb = query_guided_compressed_tokens.unsqueeze(2)
        all_compressed = F.pad(comp_emb, (0, 0, 0, effective_max_seg_len - 1), value=0.0)
        all_compressed_masks = F.pad(
            torch.ones(batch_size, num_segments, 1, dtype=torch.bool, device=device),
            (0, effective_max_seg_len - 1), value=False
        )

        all_uncompressed = all_uncompressed.view(batch_size, -1, hidden_size)
        all_compressed = all_compressed.view(batch_size, -1, hidden_size)
        all_uncompressed_masks = all_uncompressed_masks.view(batch_size, -1)
        all_compressed_masks = all_compressed_masks.view(batch_size, -1)

        # print(f"[DEBUG] Before selection:")
        # print(f"  all_uncompressed shape: {all_uncompressed.shape}")
        # print(f"  all_compressed shape: {all_compressed.shape}")
        # print(f"  all_uncompressed_masks sum: {all_uncompressed_masks.sum().item()}")
        # print(f"  all_compressed_masks sum: {all_compressed_masks.sum().item()}")

        # ==============================
        # 9. Semantic alignment (compressed part only)
        # ==============================
        fused_compressed = self.semantic_alignment_layer(all_compressed)

        # ==============================
        # 10. Dynamic selection
        # ==============================
        keep_mask_flat = original_order_keep_mask.unsqueeze(-1).expand(
            -1, -1, effective_max_seg_len
        ).reshape(batch_size, -1)
        keep_mask_for_emb = keep_mask_flat.unsqueeze(-1).expand(-1, -1, hidden_size)

        final_embeddings = torch.where(keep_mask_for_emb, all_uncompressed, fused_compressed)
        final_masks = torch.where(keep_mask_flat, all_uncompressed_masks, all_compressed_masks)

        # print(f"[DEBUG] After selection (before remove_padding):")
        # print(f"  final_embeddings shape: {final_embeddings.shape}")
        # print(f"  final_masks shape: {final_masks.shape}")
        # print(f"  final_masks sum: {final_masks.sum().item()}")

        # ==============================
        # 11. Remove padding based on the remove_padding parameter
        # ==============================
        if remove_padding:
            print(f"[DEBUG] Entering _compact_sequence...")
            final_embeddings, final_masks = self._compact_sequence(
                final_embeddings, 
                final_masks, 
                batch_size, 
                num_segments, 
                effective_max_seg_len,
                device
            )
            
            # print(f"[DEBUG] After remove_padding:")
            # print(f"  final_embeddings shape: {final_embeddings.shape}")
            # print(f"  final_masks shape: {final_masks.shape}")
            # print(f"  final_masks sum: {final_masks.sum().item()}")
            
            original_len = batch_size * num_segments * effective_max_seg_len
            compressed_len = final_masks.sum().item()
            actual_compression = 1 - (compressed_len / original_len)
            print(f"[REMOVE_PADDING] Actual compression: {actual_compression:.2%}, "
                f"Length: {original_len} -> {int(compressed_len)}")

        return final_embeddings, final_masks, contrastive_loss


    def _compact_sequence(
        self,
        embeddings: torch.Tensor,
        masks: torch.Tensor,
        batch_size: int,
        num_segments: int,
        effective_max_seg_len: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks = masks.bool()

        compact_embeddings_list = []
        compact_masks_list = []

        # print(f"\n{'='*60}")
        # print(f"[_compact_sequence DEBUG]")
        # print(f"{'='*60}")
        # print(f"Input embeddings shape: {embeddings.shape}")
        # print(f"Input masks shape: {masks.shape}")
        # print(f"Input masks sum: {masks.sum().item()}")
        # print(f"batch_size: {batch_size}")
        
        for b in range(batch_size):
            valid_mask = masks[b]  # [S*L_eff]
            
            # print(f"\nBatch {b}:")
            # print(f"  valid_mask shape: {valid_mask.shape}")
            # print(f"  valid_mask sum: {valid_mask.sum().item()}")
            # print(f"  valid_mask dtype: {valid_mask.dtype}")
            
            valid_embeds = embeddings[b][valid_mask]  # [N_valid, H]
            print(f"  valid_embeds shape after indexing: {valid_embeds.shape}")
            
            valid_attention_mask = torch.ones(valid_embeds.shape[0], dtype=torch.bool, device=device)
            print(f"  valid_attention_mask shape: {valid_attention_mask.shape}")
            
            compact_embeddings_list.append(valid_embeds)
            compact_masks_list.append(valid_attention_mask)
        
        # Padding到batch内最大长度
        print(f"\ncompact_embeddings_list lengths: {[e.shape[0] for e in compact_embeddings_list]}")
        max_valid_len = max(emb.shape[0] for emb in compact_embeddings_list)
        print(f"max_valid_len: {max_valid_len}")
        
        # 处理空序列的情况
        if max_valid_len == 0:
            max_valid_len = 1
            print(f"max_valid_len was 0, set to 1")
        
        def pad_to_len(tensor, target_len, pad_value):
            """padding到目标长度"""
            print(f"    pad_to_len called: tensor.shape={tensor.shape}, target_len={target_len}, pad_value={pad_value}")
            if tensor.shape[0] < target_len:
                pad_size = target_len - tensor.shape[0]
                if tensor.dim() == 2:  # embeddings: [N, H]
                    result = F.pad(tensor, (0, 0, 0, pad_size), value=pad_value)
                else:  # masks: [N]
                    result = F.pad(tensor, (0, pad_size), value=pad_value)
                print(f"    padded result shape: {result.shape}")
                return result
            print(f"    no padding needed")
            return tensor
        
        print(f"\nPadding embeddings...")
        compact_embeddings = torch.stack([
            pad_to_len(emb, max_valid_len, 0.0) 
            for emb in compact_embeddings_list
        ])  # [B, max_valid_len, H]
        print(f"compact_embeddings after stack: {compact_embeddings.shape}")
        
        print(f"\nPadding masks...")
        compact_masks = torch.stack([
            pad_to_len(mask.float(), max_valid_len, 0.0) 
            for mask in compact_masks_list
        ])  # [B, max_valid_len]
        # print(f"compact_masks after stack: {compact_masks.shape}")
        
        # print(f"\nFinal output:")
        # print(f"  compact_embeddings shape: {compact_embeddings.shape}")
        # print(f"  compact_masks shape: {compact_masks.shape}")
        # print(f"  compact_masks sum: {compact_masks.sum().item()}")
        # print(f"{'='*60}\n")
        
        return compact_embeddings, compact_masks

    def normalize(
        self,
        x
    ):
        # x.shape: [seq_len, hidden_size]
        max_val = torch.max(x)
        min_val = torch.min(x)
        
        normalized_x = (x - min_val) / (max_val - min_val + 1e-8)
        return self.scale * normalized_x
    

    def split_to_segments(
        self,
        enc_doc_ids,
        enc_doc_mask,
        segment_size
    ):
        num_segments = math.ceil(1.0 * enc_doc_ids.shape[1] / segment_size)
        if num_segments == 1:
            segments = [enc_doc_ids]
            segments_masks = [enc_doc_mask]
        else:
            segments = [enc_doc_ids[:, i * segment_size : (i + 1) * segment_size] if i != num_segments - 1 else \
                enc_doc_ids[:, i * segment_size : ] for i in range(num_segments)]
            segments_masks = [enc_doc_mask[:, i * segment_size : (i + 1) * segment_size] if i != num_segments - 1 else \
                enc_doc_mask[:, i * segment_size : ] for i in range(num_segments)]
            
        return segments, segments_masks

    def generate_post_append_memorys(
        self,
        input_ids,
        input_mask
    ):
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        mem_map = {
            4: 128,
            8: 64
        }
        
        if self.is_random:
            actual_mem_size = self.generate_post_append_size()
        else:
            if not self.icae_infer:
                actual_mem_size = self.mem_size
            else:
                actual_mem_size = mem_map[self.merge_size]
            
        self.memory_sequence = self.memory_sequence.to(device)
        
        memorys_embeddings = torch.cat([self.memory_token_embed(self.memory_sequence[:actual_mem_size] - self.vocab_size).unsqueeze(0) for _ in range(batch_size)], dim=0)
        memorys_attention_mask = torch.ones(memorys_embeddings.shape[0], memorys_embeddings.shape[1]).to(device)
        
        input_embeddings = self.llm_encoder.get_base_model().model.embed_tokens(input_ids)
        
        final_input_embeddings = torch.cat((input_embeddings, memorys_embeddings), dim=1)
        attention_mask = torch.cat((input_mask, memorys_attention_mask), dim=1)
        
        last_hidden_state = self.llm_encoder(
            inputs_embeds=final_input_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True
        ).hidden_states[-1]
        
        final_memorys = last_hidden_state[:, -actual_mem_size:, :]
        # aligned_memorys = final_memorys.to(torch.bfloat16)
        aligned_memorys = self.semantic_alignment_layer(final_memorys).to(torch.bfloat16)
        
        return aligned_memorys, memorys_attention_mask
    
    def pad_embedds_across_gpus(self, aligned_memorys, memorys_mask):
        local_seq_len = aligned_memorys.shape[1]
        local_seq_len_tensor = torch.tensor([local_seq_len], dtype=torch.long, device=aligned_memorys.device)

        world_size = dist.get_world_size()
        gathered_seq_lens = torch.zeros(world_size, dtype=torch.long, device=aligned_memorys.device)
        dist.all_gather_into_tensor(gathered_seq_lens, local_seq_len_tensor)
        max_seq_len = gathered_seq_lens.max().item()

        if local_seq_len < max_seq_len:
            pad_len = max_seq_len - local_seq_len
            aligned_memorys = torch.nn.functional.pad(aligned_memorys, (0, 0, 0, pad_len), value=0.0)
            memorys_mask = torch.nn.functional.pad(memorys_mask, (0, pad_len), value=False)

        return aligned_memorys, memorys_mask
    
    def pad_ids_across_gpus(self, input_ids, input_mask):
        local_seq_len = input_ids.shape[1]
        local_seq_len_tensor = torch.tensor([local_seq_len], dtype=torch.long, device=input_ids.device)

        world_size = dist.get_world_size()
        gathered_seq_lens = torch.zeros(world_size, dtype=torch.long, device=input_ids.device)
        dist.all_gather_into_tensor(gathered_seq_lens, local_seq_len_tensor)
        max_seq_len = gathered_seq_lens.max().item()

        if local_seq_len < max_seq_len:
            pad_len = max_seq_len - local_seq_len
            # 注意：value=0 是整数，保持 input_ids 的 dtype 为 long
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)
            input_mask = torch.nn.functional.pad(input_mask, (0, pad_len), value=False)

        return input_ids, input_mask

    def forward(
        self,
        enc_doc_ids,
        enc_doc_mask,
        target_doc_ids,
        target_doc_mask,
        llm_ins_ids=None,
        llm_ins_mask=None,
        enc_prefix_ids=None,
        enc_prefix_mask=None,
        enc_repeat_ids=None,
        enc_repeat_mask=None,
        llm_answer_ids=None,
        llm_answer_mask=None,
        llm_que_ids=None,
        llm_que_mask=None,
        enc_continue_ids=None,
        enc_continue_mask=None,
        **kwargs
    ):

        if self.split:
            segments, segments_mask = self.split_to_segments(
                enc_doc_ids, 
                enc_doc_mask, 
                self.training_args.segment_size
            )
            segments_merge_size = [self.generate_merge_size() for _ in range(len(segments))]
        else:
            segments = [enc_doc_ids]
            segments_mask = [enc_doc_mask]

        if self.enable_encore:
            aligned_memorys, memorys_mask, contrastive_loss = self.generate_ram_memorys(
                input_ids_list=segments,
                input_mask_list=segments_mask,
                query_ids=kwargs['enc_que_ids'],
                query_input_mask=kwargs['enc_que_mask'],
                accumulation_ratio=self.top_p,
                compute_contrastive=self.use_contrastive_loss and self.training,
                pos_seg_ids=kwargs["pos_seg_ids"],
            )
        else:
            if self.post_append:
                aligned_memorys, memorys_masks = zip(*[
                    self.generate_post_append_memorys(input_ids=s, input_mask=m)
                    for s, m in zip(segments, segments_mask)
                ])     
            else:
                if self.split:
                    if self.autoregressive:
                        aligned_memorys = None
                        memorys_mask = None
                        for idx, (s, m, m_s) in enumerate(zip(segments, segments_mask, segments_merge_size)):
                            aligned_memorys, memorys_mask = self.generate_autoregressive_tkdr_memorys(
                                input_ids=s, 
                                input_mask=m,
                                merge_size=m_s,
                                query_ids=kwargs['enc_que_ids'],
                                query_input_mask=kwargs['enc_que_mask'],
                                pre_mem_embeds=aligned_memorys,
                                pre_mem_attention_mask=memorys_mask,
                                end_flag=(idx == len(segments) - 1)
                            ) 
                    else:
                        if self.launch_tkdr:
                            aligned_memorys, memorys_masks = zip(*[
                                self.generate_tkdr_memorys(
                                    input_ids=s, 
                                    input_mask=m,
                                    merge_size=m_s,
                                    query_ids=kwargs['enc_que_ids'],
                                    query_mask=kwargs['enc_que_mask']
                                )
                                for s, m, m_s in zip(segments, segments_mask, segments_merge_size)
                            ])
                        else:
                            aligned_memorys, memorys_masks = zip(*[
                                self.generate_pooling_memorys(
                                    input_ids=s, 
                                    input_mask=m,
                                    merge_size=m_s
                                )
                                for s, m, m_s in zip(segments, segments_mask, segments_merge_size)
                            ])
                else:
                    if self.keft:
                        aligned_memorys, memorys_masks = zip(*[
                            self.generate_query_guided_pooling_memorys(
                                input_ids=s, 
                                input_mask=m, 
                                query_ids=kwargs['enc_que_ids'],
                                query_mask=kwargs['enc_que_mask'],
                                merge_size=self.merge_size
                            )
                            for s, m in zip(segments, segments_mask)
                        ])
                    else:
                        aligned_memorys, memorys_masks = zip(*[
                            self.generate_pooling_memorys(
                                input_ids=s, 
                                input_mask=m, 
                                merge_size=self.merge_size
                            )
                            for s, m in zip(segments, segments_mask)
                        ])

            if not self.autoregressive:
                if len(aligned_memorys) == 1:
                    aligned_memorys = aligned_memorys[0]
                    memorys_mask = memorys_masks[0]
                else:
                    aligned_memorys = torch.cat(aligned_memorys, dim=1)
                    memorys_mask = torch.cat(memorys_masks, dim=1)

        if not self.empty:
            llm_ins_embeds = self.llm.model.embed_tokens(llm_ins_ids)
            llm_que_embeds = self.llm.model.embed_tokens(llm_que_ids)
            answer_embeds = self.llm.model.embed_tokens(llm_answer_ids)

            llm_input_embedings = torch.cat((llm_ins_embeds, aligned_memorys, \
                llm_que_embeds, answer_embeds), dim=1)
            llm_attention_mask = torch.cat((llm_ins_mask, memorys_mask, \
                llm_que_mask, llm_answer_mask), dim=1)
        else:
            llm_que_embeds = self.llm.model.embed_tokens(llm_que_ids)
            answer_embeds = self.llm.model.embed_tokens(llm_answer_ids)

            llm_input_embedings = torch.cat((aligned_memorys, \
                llm_que_embeds, answer_embeds), dim=1)
            llm_attention_mask = torch.cat((memorys_mask, \
                llm_que_mask, llm_answer_mask), dim=1)
            
        llm_fine_tune_labels = torch.full_like(llm_attention_mask, -100)
        llm_fine_tune_labels[:, -llm_answer_ids.size(1):] = llm_answer_ids.masked_fill(
            ~llm_answer_mask.bool(), -100,
        )
        
        llm_outputs = self.llm(
            inputs_embeds=llm_input_embedings,
            attention_mask=llm_attention_mask
        )  
        # pprint(llm_input_embedings.shape)

        logits = llm_outputs.logits

        effective_logits = logits[:, :-1,:].reshape(-1, logits.size(-1))
        target_ids = llm_fine_tune_labels[:, 1:].reshape(-1).to(torch.long)
        # print(f"llm_input_embedings.shape: {llm_input_embedings.shape}")
        # print("Max target ID:", target_ids.max().item())
        # print("LLM Vocab Size:", self.llm.config.vocab_size)
        # assert target_ids.max() < self.llm.config.vocab_size, "Target ID out of bounds!"
        main_loss = self.loss_fct(effective_logits, target_ids)

        # 决定是否加入对比损失
        total_loss = main_loss
        if self.use_contrastive_loss and contrastive_loss is not None:
            total_loss = total_loss + contrastive_loss

        return {"loss" : total_loss, "logits" : logits}

