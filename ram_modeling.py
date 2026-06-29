import os
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, TrainingArguments

from data_utils import DEFAULT_NQ_PATH

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    compression_rate: int = field(
        default=16,
        metadata={"help": "target compression rate (16 or 32)"},
    )
    use_contrastive_loss: bool = field(
        default=True,
        metadata={"help": "contrastive loss (only for training)"},
    )
    contrastive_temperature: float = field(default=0.1)
    contrastive_loss_weight: float = field(default=1.0)
    keep_ratio: float = field(
        default=0.25,
        metadata={"help": "Fraction of relevant segments kept as original tokens."},
    )
    use_flash_attention_2: bool = field(default=False)
    use_transform_layer: bool = field(
        default=False,
        metadata={
            "help": "memory_fusion_layer apply to compressed group vectors.",
        },
    )
    num_mem_fusion_layers: int = field(default=1)
    lora_r: int = field(default=0)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    train: bool = field(default=False, metadata={"help": "True for training initialization; False for inference"})
    encoder_layers: int = field(default=8, metadata={"help": "Number of encoder layers"})

    def __post_init__(self):
        if self.compression_rate not in (16, 32):
            raise ValueError("compression_rate must be either 16 or 32")
        if not 0.0 <= self.keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be between 0 and 1")
        if self.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be positive")


@dataclass
class DataArguments:
    train_file: str = field(
        default=DEFAULT_NQ_PATH,
        metadata={"help": "Training json/jsonl"},
    )
    test_file: str = field(
        default=DEFAULT_NQ_PATH,
        metadata={"help": "Test json/jsonl"},
    )
    debug_data: bool = field(default=False)
    test_sample_index: int = field(default=0)
    num_samples: int = field(default=2655)


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=28000)
    report_to: Optional[str] = field(default="none")
    project_name: Optional[str] = field(default="ram_nq")
    max_steps: int = field(default=10000)
    save_strategy: Optional[str] = field(default="steps")
    save_steps: int = field(default=10000)
    eval_strategy: Optional[str] = field(default="steps")
    eval_steps: int = field(default=20000)
    num_train_epochs: int = field(default=1)
    add_special_token_for_lm: bool = field(default=False)
    restore_from: str = field(default="")
    overwrite_output_dir: bool = field(default=True)
    logging_steps: int = field(default=200)
    deepspeed: str = field(default="")
    bf16: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=1)
    optim: str = field(default="adamw_torch")
    per_device_train_batch_size: int = field(default=1)
    lr_scheduler_type: str = field(default="cosine")
    learning_rate: float = field(default=1e-5)
    gradient_checkpointing: bool = field(default=True)
    warmup_ratio: float = field(default=0.1)
    weight_decay: float = field(default=0.01)
    seed: int = field(default=42)


def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_p = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable} || all params: {all_p} || trainable%: {100 * trainable / all_p:.2f}")


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False




class RAM(nn.Module):
    def __init__(self, model_args: ModelArguments, training_args: TrainingArguments, tokenizer):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        
        dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        encoder_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        encoder_config.num_hidden_layers = model_args.encoder_layers
        self.encoder = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=encoder_config,
            cache_dir=training_args.cache_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.decoder = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        
        if model_args.lora_r > 0:
            if LoraConfig is None or get_peft_model is None:
                raise ImportError("PEFT is required when lora_r is greater than zero")
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                target_modules="all-linear",
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.decoder = get_peft_model(self.decoder, lora_config)
            
            print_trainable_parameters(self.encoder)
            print_trainable_parameters(self.decoder)
        
        self.compression_rate = model_args.compression_rate
        self.keep_ratio = model_args.keep_ratio
        self.use_contrastive_loss = model_args.use_contrastive_loss
        self.contrastive_temperature = model_args.contrastive_temperature
        self.contrastive_loss_weight = model_args.contrastive_loss_weight
        self.use_transform_layer = model_args.use_transform_layer

        self.dim = self.encoder.config.hidden_size
        decoder_hidden_size = self.decoder.config.hidden_size
        self.semantic_alignment_layer = nn.Linear(self.dim, decoder_hidden_size).to(dtype=dtype)

        if self.use_transform_layer:
            fusion_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            fusion_config.num_hidden_layers = model_args.num_mem_fusion_layers
            self.memory_fusion_layer = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=fusion_config,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            self.memory_fusion_layer.config.use_cache = False
            self.memory_fusion_layer.use_cache = False

        self._configure_trainability()
        print_trainable_parameters(self.encoder)
        print_trainable_parameters(self.decoder)
        
        if model_args.train and training_args.restore_from:
            self.init_training()

    def _configure_trainability(self):
        if not self.model_args.train:
            self.encoder.eval()
            self.decoder.eval()
            if self.use_transform_layer and hasattr(self, "memory_fusion_layer"):
                self.memory_fusion_layer.eval()
            return

        for param in self.decoder.parameters():
            param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.semantic_alignment_layer.parameters():
            param.requires_grad = True
        if self.use_transform_layer and hasattr(self, "memory_fusion_layer"):
            for param in self.memory_fusion_layer.parameters():
                param.requires_grad = True

    def init_training(self):
        print_trainable_parameters(self)
        restore_from = self.training_args.restore_from
        print(f"Loading checkpoint: {restore_from}...")
        if os.path.isdir(restore_from):
            candidates = [
                os.path.join(restore_from, "model.safetensors"),
                os.path.join(restore_from, "pytorch_model.bin"),
            ]
            ckpt = next((p for p in candidates if os.path.exists(p)), None)
        else:
            ckpt = restore_from
            
        if ckpt:
            raw = load_file(ckpt) if ckpt.endswith(".safetensors") else torch.load(ckpt, map_location="cpu")
            state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded from {ckpt}")

        # for mod in (self.encoder, self.decoder):
        #     if any(p.requires_grad for p in mod.parameters()):
        #         mod.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if self.use_transform_layer and hasattr(self, "memory_fusion_layer"):
            self.memory_fusion_layer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def _select_keep_mask(self, segment_scores, segment_mask):
        keep_mask = torch.zeros_like(segment_mask, dtype=torch.bool)
        for batch_index in range(segment_scores.size(0)):
            valid_indices = segment_mask[batch_index].nonzero(as_tuple=True)[0]
            num_kept = int(valid_indices.numel() * self.keep_ratio)
            if num_kept == 0:
                continue
            local_scores = segment_scores[batch_index, valid_indices]
            selected = valid_indices[torch.topk(local_scores, k=num_kept).indices]
            keep_mask[batch_index, selected] = True
        return keep_mask

    def _assemble_hybrid_memory(
        self,
        original_embeddings,
        compressed_embeddings,
        token_mask,
        segment_mask,
        keep_mask,
    ):
        sequences = []
        masks = []
        for batch_index in range(original_embeddings.size(0)):
            pieces = []
            valid_segments = segment_mask[batch_index].nonzero(as_tuple=True)[0]
            for segment_index in valid_segments.tolist():
                if keep_mask[batch_index, segment_index]:
                    pieces.append(
                        original_embeddings[batch_index, segment_index][
                            token_mask[batch_index, segment_index]
                        ]
                    )
                else:
                    pieces.append(
                        compressed_embeddings[batch_index, segment_index].unsqueeze(0)
                    )
            if not pieces:
                raise ValueError("Context contains no valid segments")
            sequence = torch.cat(pieces, dim=0)
            sequences.append(sequence)
            masks.append(
                torch.ones(sequence.size(0), dtype=torch.bool, device=sequence.device)
            )
        return (
            nn.utils.rnn.pad_sequence(sequences, batch_first=True),
            nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=False),
        )

    def _compute_contrastive_loss(
        self, segment_scores, segment_mask, positive_segment_mask
    ):
        positives = positive_segment_mask.bool() & segment_mask
        valid_rows = positives.any(dim=1)
        if not valid_rows.any():
            return segment_scores.new_zeros(())
        logits = segment_scores / self.contrastive_temperature
        valid_logits = logits.masked_fill(~segment_mask, -torch.inf)
        positive_logits = logits.masked_fill(~positives, -torch.inf)
        losses = torch.logsumexp(valid_logits, dim=1) - torch.logsumexp(
            positive_logits, dim=1
        )
        return losses[valid_rows].mean()

    def _combine_training_losses(self, qa_loss, contrastive_loss):
        if not self.use_contrastive_loss:
            return qa_loss
        return qa_loss + self.contrastive_loss_weight * contrastive_loss

    def _split_context(self, context_ids, context_mask):
        pad_length = (-context_ids.size(1)) % self.compression_rate
        if pad_length:
            context_ids = F.pad(
                context_ids,
                (0, pad_length),
                value=self.tokenizer.pad_token_id,
            )
            context_mask = F.pad(context_mask, (0, pad_length), value=0)
        segment_ids = context_ids.view(
            context_ids.size(0), -1, self.compression_rate
        )
        token_mask = context_mask.view(
            context_mask.size(0), -1, self.compression_rate
        ).bool()
        return segment_ids, token_mask, token_mask.any(dim=-1)

    def _align_compressed_memory(self, compressed, low_segment_mask):
        aligned = self.semantic_alignment_layer(compressed)
        if not self.use_transform_layer or not low_segment_mask.any():
            return aligned

        batch_size, _, hidden_size = compressed.shape
        max_low = int(low_segment_mask.sum(dim=1).max().item())
        packed = compressed.new_zeros(batch_size, max_low, hidden_size)
        packed_mask = torch.zeros(
            batch_size,
            max_low,
            dtype=torch.bool,
            device=compressed.device,
        )
        low_indices = []
        for batch_index in range(batch_size):
            indices = low_segment_mask[batch_index].nonzero(as_tuple=True)[0]
            low_indices.append(indices)
            if indices.numel():
                packed[batch_index, : indices.numel()] = compressed[
                    batch_index, indices
                ]
                packed_mask[batch_index, : indices.numel()] = True

        safe_mask = packed_mask.clone()
        safe_mask[~safe_mask.any(dim=1), 0] = True
        fused = self.memory_fusion_layer(
            inputs_embeds=packed,
            attention_mask=safe_mask.long(),
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]
        fused = self.semantic_alignment_layer(fused)
        for batch_index, indices in enumerate(low_indices):
            if indices.numel():
                aligned[batch_index, indices] = fused[
                    batch_index, : indices.numel()
                ]
        return aligned

    def generate_encore_memorys(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        positive_segment_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        segment_ids, token_mask, segment_mask = self._split_context(
            context_ids, context_mask
        )
        batch_size, num_segments, segment_width = segment_ids.shape
        common_width = max(query_ids.size(1), segment_width)

        padded_query_ids = F.pad(
            query_ids,
            (0, common_width - query_ids.size(1)),
            value=self.tokenizer.pad_token_id,
        )
        padded_query_mask = F.pad(
            query_mask, (0, common_width - query_mask.size(1)), value=0
        ).bool()
        padded_segment_ids = F.pad(
            segment_ids,
            (0, common_width - segment_width),
            value=self.tokenizer.pad_token_id,
        )
        padded_token_mask = F.pad(
            token_mask, (0, common_width - segment_width), value=False
        )

        encoder_ids = torch.cat(
            [padded_query_ids.unsqueeze(1), padded_segment_ids], dim=1
        )
        encoder_mask = torch.cat(
            [padded_query_mask.unsqueeze(1), padded_token_mask], dim=1
        )
        hidden = self.encoder(
            input_ids=encoder_ids.reshape(-1, common_width),
            attention_mask=encoder_mask.reshape(-1, common_width),
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]
        hidden = hidden.view(batch_size, num_segments + 1, common_width, -1)

        query_hidden = hidden[:, 0]
        segment_hidden = hidden[:, 1:, :segment_width]
        query_weights = padded_query_mask.unsqueeze(-1).to(query_hidden.dtype)
        query_repr = (query_hidden * query_weights).sum(dim=1) / query_weights.sum(
            dim=1
        ).clamp_min(1)
        segment_weights = token_mask.unsqueeze(-1).to(segment_hidden.dtype)
        segment_repr = (segment_hidden * segment_weights).sum(
            dim=2
        ) / segment_weights.sum(dim=2).clamp_min(1)

        segment_scores = F.cosine_similarity(
            segment_repr, query_repr.unsqueeze(1), dim=-1
        )
        segment_scores = segment_scores.masked_fill(~segment_mask, -torch.inf)
        keep_mask = self._select_keep_mask(segment_scores, segment_mask)

        token_scores = F.cosine_similarity(
            segment_hidden, query_repr[:, None, None, :], dim=-1
        )
        token_scores = token_scores.masked_fill(
            ~token_mask, torch.finfo(token_scores.dtype).min
        )
        token_weights = F.softmax(token_scores, dim=-1) * token_mask.to(
            token_scores.dtype
        )
        token_weights = token_weights / token_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        compressed = (segment_hidden * token_weights.unsqueeze(-1)).sum(dim=2)
        low_segment_mask = segment_mask & ~keep_mask
        aligned_compressed = compressed
        aligned_compressed = self._align_compressed_memory(
            compressed, low_segment_mask
        )

        original_embeddings = self.decoder.get_input_embeddings()(segment_ids)
        memory, memory_mask = self._assemble_hybrid_memory(
            original_embeddings,
            aligned_compressed,
            token_mask,
            segment_mask,
            keep_mask,
        )

        contrastive_loss = segment_scores.new_zeros(())
        if self.training and self.use_contrastive_loss:
            if positive_segment_mask is None:
                positive_segment_mask = torch.zeros_like(segment_mask)
            positive_segment_mask = positive_segment_mask[:, :num_segments]
            positive_segment_mask = F.pad(
                positive_segment_mask,
                (0, num_segments - positive_segment_mask.size(1)),
                value=False,
            )
            contrastive_loss = self._compute_contrastive_loss(
                segment_scores, segment_mask, positive_segment_mask
            )
        return memory, memory_mask, contrastive_loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        query_ids=None,
        query_attention_mask=None,
        decoder_query_ids=None,
        decoder_query_attention_mask=None,
        positive_segment_mask=None,
        labels=None,
        **kwargs,
    ):
        if query_ids is None or query_attention_mask is None:
            raise ValueError("query_ids and query_attention_mask are required")

        mem, mem_mask, contrastive_loss = self.generate_encore_memorys(
            input_ids,
            attention_mask,
            query_ids,
            query_attention_mask,
            positive_segment_mask,
        )

        if decoder_query_ids is None:
            decoder_query_ids = query_ids
        if decoder_query_attention_mask is None:
            decoder_query_attention_mask = query_attention_mask

        dec_dtype = self.decoder.get_input_embeddings().weight.dtype
        mem = mem.to(dec_dtype)
        q_emb = self.decoder.get_input_embeddings()(decoder_query_ids).to(dec_dtype)
        q_mask = decoder_query_attention_mask.bool()

        if labels is not None:
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = self.tokenizer.pad_token_id or 0
            label_embedding = self.decoder.get_input_embeddings()(safe_labels).to(dec_dtype)
            full_embedding = torch.cat([mem, q_emb, label_embedding], dim=1)
            full_mask = torch.cat([mem_mask.bool(), q_mask, labels != -100], dim=1)
            ignore_mask = torch.full(
                mem_mask.shape, -100, device=labels.device, dtype=labels.dtype
            )
            ignore_query = torch.full(
                decoder_query_ids.shape,
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            full_label = torch.cat([ignore_mask, ignore_query, labels], dim=1)
            out = self.decoder(
                inputs_embeds=full_embedding,
                attention_mask=full_mask,
                labels=full_label,
                return_dict=True,
            )
            return {
                "loss": self._combine_training_losses(out.loss, contrastive_loss),
                "qa_loss": out.loss.detach(),
                "contrastive_loss": contrastive_loss.detach(),
            }

        decoder_input = torch.cat([mem, q_emb], dim=1)
        decoder_mask = torch.cat([mem_mask, q_mask], dim=1).long()
        return self.decoder.generate(
            inputs_embeds=decoder_input,
            attention_mask=decoder_mask,
            max_new_tokens=100,
            min_new_tokens=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            use_cache=True,
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.encoder.gradient_checkpointing_enable(*args, **kwargs)
        self.decoder.gradient_checkpointing_enable(*args, **kwargs)
        if self.use_transform_layer and hasattr(self, "memory_fusion_layer"):
            self.memory_fusion_layer.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        self.encoder.gradient_checkpointing_disable(*args, **kwargs)
        self.decoder.gradient_checkpointing_disable(*args, **kwargs)
        if self.use_transform_layer and hasattr(self, "memory_fusion_layer"):
            self.memory_fusion_layer.gradient_checkpointing_disable(*args, **kwargs)
