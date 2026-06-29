import json
import os

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    SequentialSampler,
)
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from data_utils import prepare_nq_example
from eval_utils import compute_exact, compute_f1
from ram_modeling import DataArguments, ModelArguments, RAM, TrainingArguments


class JsonlDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    rank = dist.get_rank() if distributed else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return distributed, rank, world_size, device


def load_checkpoint(model, restore_from):
    if not restore_from:
        return
    checkpoint = restore_from
    if os.path.isdir(checkpoint):
        candidates = [
            os.path.join(checkpoint, "model.safetensors"),
            os.path.join(checkpoint, "pytorch_model.bin"),
        ]
        checkpoint = next((path for path in candidates if os.path.exists(path)), "")
    if not checkpoint:
        raise FileNotFoundError(f"No model checkpoint found under {restore_from}")

    state = (
        load_file(checkpoint)
        if checkpoint.endswith(".safetensors")
        else torch.load(checkpoint, map_location="cpu")
    )
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {
        (key.replace("icae.", "encoder.", 1) if key.startswith("icae.") else key): value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(state, strict=False)
    print(
        f"Loaded {checkpoint}: missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )


def load_nq_rows(test_file, num_samples):
    if num_samples == 0:
        return []
    rows = []
    with open(test_file, encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            sample = prepare_nq_example(json.loads(line), sample_index=line_index)
            sample["sample_index"] = line_index
            rows.append(sample)
            if num_samples >= 0 and len(rows) >= num_samples:
                break
    return rows


def collate_batch(rows, tokenizer, max_length):
    contexts = tokenizer(
        [row["context"] for row in rows],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    queries = tokenizer(
        [row["question"] for row in rows],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    decoder_queries = tokenizer(
        [row["decoder_prompt"] for row in rows],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    return {
        "input_ids": contexts["input_ids"],
        "attention_mask": contexts["attention_mask"],
        "query_ids": queries["input_ids"],
        "query_attention_mask": queries["attention_mask"],
        "decoder_query_ids": decoder_queries["input_ids"],
        "decoder_query_attention_mask": decoder_queries["attention_mask"],
        "sample_indices": [row["sample_index"] for row in rows],
        "questions": [row["question"] for row in rows],
        "answers": [row["answers"] for row in rows],
    }


def compute_metrics(results):
    enriched = []
    for result in results:
        prediction = result["prediction"]
        answers = result["answers"]
        em = max(compute_exact(answer, prediction) for answer in answers)
        f1 = max(compute_f1(prediction, answer) for answer in answers)
        enriched.append({**result, "em": em, "f1": f1})
    count = len(enriched)
    summary = {
        "total_samples": count,
        "avg_em": sum(row["em"] for row in enriched) / count if count else 0.0,
        "avg_f1": sum(row["f1"] for row in enriched) / count if count else 0.0,
    }
    return summary, enriched


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    distributed, rank, world_size, device = setup_distributed()
    model_args.train = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = RAM(model_args, training_args, tokenizer)
    load_checkpoint(model, training_args.restore_from)
    model.to(device).eval()

    dataset = JsonlDataset(
        load_nq_rows(data_args.test_file, data_args.num_samples)
    )
    sampler = (
        DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if distributed
        else SequentialSampler(dataset)
    )
    loader = DataLoader(
        dataset,
        batch_size=training_args.per_device_eval_batch_size,
        sampler=sampler,
        collate_fn=lambda rows: collate_batch(
            rows, tokenizer, training_args.model_max_length
        ),
    )

    local_results = []
    with torch.no_grad():
        for batch in tqdm(loader, disable=rank != 0):
            generated = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                query_ids=batch["query_ids"].to(device),
                query_attention_mask=batch["query_attention_mask"].to(device),
                decoder_query_ids=batch["decoder_query_ids"].to(device),
                decoder_query_attention_mask=batch[
                    "decoder_query_attention_mask"
                ].to(device),
            )
            predictions = tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )
            local_results.extend(
                {
                    "sample_index": sample_index,
                    "question": question,
                    "prediction": prediction.strip(),
                    "answers": answers,
                }
                for sample_index, question, prediction, answers in zip(
                    batch["sample_indices"],
                    batch["questions"],
                    predictions,
                    batch["answers"],
                )
            )

    if distributed:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_results)
        results = [row for shard in gathered for row in shard]
    else:
        results = local_results

    if rank == 0:
        results = list(
            {
                row["sample_index"]: row for row in sorted(
                    results, key=lambda item: item["sample_index"]
                )
            }.values()
        )
        for row in results:
            row.pop("sample_index", None)
        metrics, results = compute_metrics(results)
        os.makedirs(training_args.output_dir, exist_ok=True)
        tag = (
            f"c{model_args.compression_rate}_k{model_args.keep_ratio}_{len(results)}"
        )
        result_path = os.path.join(
            training_args.output_dir, f"nq_inference_results_{tag}.jsonl"
        )
        metrics_path = os.path.join(
            training_args.output_dir, f"nq_inference_metrics_{tag}.json"
        )
        with open(result_path, "w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        print(json.dumps(metrics, ensure_ascii=False))

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
