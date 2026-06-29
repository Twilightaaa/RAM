import os

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint

from data_utils import build_positive_segment_mask, prepare_nq_example


class InstructFTTokenizeFunction:
    def __init__(
        self,
        tokenizer,
        max_length,
        segment_size,
        use_contrastive: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.segment_size = segment_size
        self.use_contrastive = use_contrastive

    def __call__(self, examples):
        output = {
            "input_ids": [],
            "query_ids": [],
            "decoder_query_ids": [],
            "labels": [],
            "positive_segment_mask": [],
        }
        for index in range(len(examples["question"])):
            sample = prepare_nq_example(
                {key: examples[key][index] for key in ("question", "ctxs", "answers")},
                sample_index=index,
            )
            context_ids = self.tokenizer.encode(
                sample["context"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )
            output["input_ids"].append(context_ids)
            output["query_ids"].append(
                self.tokenizer.encode(sample["question"], add_special_tokens=False)
            )
            output["decoder_query_ids"].append(
                self.tokenizer.encode(sample["decoder_prompt"], add_special_tokens=False)
            )
            output["labels"].append(
                self.tokenizer.encode(
                    f"{sample['answer']}{self.tokenizer.eos_token}",
                    add_special_tokens=False,
                )
            )
            output["positive_segment_mask"].append(
                build_positive_segment_mask(
                    context_ids,
                    sample["answers"],
                    self.tokenizer,
                    self.segment_size,
                )
                if self.use_contrastive
                else []
            )
        return output


class DataCollatorForDynamicPadding:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        def pad_ids(name, padding_value):
            values = [torch.tensor(feature[name], dtype=torch.long) for feature in features]
            padded = pad_sequence(values, batch_first=True, padding_value=padding_value)
            masks = pad_sequence(
                [torch.ones(len(value), dtype=torch.long) for value in values],
                batch_first=True,
                padding_value=0,
            )
            return padded, masks

        batch_input_ids, attention_mask = pad_ids("input_ids", self.pad_token_id)
        batch_query_ids, query_attention_mask = pad_ids("query_ids", self.pad_token_id)
        batch_decoder_query_ids, decoder_query_attention_mask = pad_ids(
            "decoder_query_ids", self.pad_token_id
        )
        batch_labels, _ = pad_ids("labels", -100)
        positive_segment_mask = pad_sequence(
            [
                torch.tensor(feature["positive_segment_mask"], dtype=torch.bool)
                for feature in features
            ],
            batch_first=True,
            padding_value=False,
        )

        return {
            "input_ids": batch_input_ids,
            "attention_mask": attention_mask,
            "query_ids": batch_query_ids,
            "query_attention_mask": query_attention_mask,
            "decoder_query_ids": batch_decoder_query_ids,
            "decoder_query_attention_mask": decoder_query_attention_mask,
            "labels": batch_labels,
            "positive_segment_mask": positive_segment_mask,
        }


def train_model(model, train_dataset, eval_dataset, training_args, tokenizer):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if local_rank == 0:
        print(training_args)
    training_args.remove_unused_columns = False

    data_collator = DataCollatorForDynamicPadding(tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        print(f"Loaded from the checkpoint: {checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
