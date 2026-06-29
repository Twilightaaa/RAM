import transformers
from datasets import load_dataset
from transformers import AutoTokenizer

from ram_modeling import DataArguments, ModelArguments, RAM, TrainingArguments
from training_utils import InstructFTTokenizeFunction, train_model

try:
    import wandb
except ImportError:
    wandb = None


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    print(
        "RAM:",
        {
            "compression_rate": model_args.compression_rate,
            "keep_ratio": model_args.keep_ratio,
            "use_contrastive_loss": model_args.use_contrastive_loss,
            "contrastive_loss_weight": model_args.contrastive_loss_weight,
            "use_transform_layer": model_args.use_transform_layer,
            "num_mem_fusion_layers": model_args.num_mem_fusion_layers,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        data_files={"train": data_args.train_file, "eval": data_args.test_file},
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]
    if data_args.debug_data:
        train_dataset = train_dataset.select(range(min(32, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(32, len(eval_dataset))))

    report_targets = training_args.report_to
    if isinstance(report_targets, str):
        report_targets = [report_targets]
    if training_args.local_rank <= 0 and "wandb" in report_targets:
        if wandb is None:
            raise ImportError("wandb is required when --report_to includes wandb")
        run_name = (
            f"ram_nq_c{model_args.compression_rate}"
            f"_k{model_args.keep_ratio}_step{training_args.max_steps}"
        )
        if data_args.debug_data:
            run_name += "_debug"
        wandb.init(project=training_args.project_name, name=run_name)

    print(f"Dataset size: train={len(train_dataset)}, eval={len(eval_dataset)}")
    tokenize_fn = InstructFTTokenizeFunction(
        tokenizer=tokenizer,
        max_length=training_args.model_max_length,
        segment_size=model_args.compression_rate,
        use_contrastive=model_args.use_contrastive_loss,
    )
    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
    )
    eval_dataset = eval_dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False,
    )

    model = RAM(model_args, training_args, tokenizer)
    train_model(model, train_dataset, eval_dataset, training_args, tokenizer)


if __name__ == "__main__":
    main()
