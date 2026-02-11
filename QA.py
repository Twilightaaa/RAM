import os
import jsonlines
# from tqdm.auto import tqdm
from tqdm import tqdm
import math
import random
import numpy as np

import torch
# from peft import (
#     LoraConfig,
# )
import transformers
from accelerate import Accelerator
# from bert_modeling import Bertencoder, ModelArguments, TrainingArguments, QGCArguments
from llm_modeling_revised import LLMCompressor, RAMArguments, ModelArguments, TrainingArguments
# from constant import *
# from logger import get_logger
from QA_dataset import InferDataset

from metrics import benchmark_function_map
import json
from QA_dataset import TrainDataset
import time

from safetensors.torch import load_file

from chat import apply_chat_template

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ===== 在这里配置 icecream，移除所有 ANSI 码 =====
from icecream import ic
import re
import sys

_ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def _clean_output(s):
    """移除 ANSI 转义码"""
    clean = _ansi_escape.sub('', s)
    print(clean, file=sys.stderr, flush=True)

ic.configureOutput(outputFunction=_clean_output)

# 使用别名
pprint = ic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset2maxgen = {
    'NQ': 100,
    'HQA': 32,
    'TQA': 32,
    'MQA': 100,
    'NARRA': 128,
    'GOV': 512,
    'NEWS': 512,
    'QSUM': 512,
    'NYC': 32,
    'CA': 32,
    'TKY': 32
}

# logger = get_logger(__name__)

tsne = TSNE(n_components=3, perplexity=30, random_state=42, max_iter=1000)

def write_jsonl(file_path, obj):
    with open(file_path, "a") as f:
        json.dump(obj, f)
        f.write("\n")

def get_full_pytorth_memory_mb():
    mem_info = {"cpu_mem_mb": 0.0, "gpu_mem_mb": 0.0, "gpu_mem_reserved_mb": 0.0}

    # 获取GPU显存
    if torch.cuda.is_available():
        mem_info["gpu_mem_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        mem_info["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        
    return mem_info

def split_to_segments(
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

def write_line(path, d):
    with open(path, 'a') as f:
        json_str = json.dumps(d)
        f.write(json_str + '\n')
      
def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def benchmark_step(
    model, 
    inputs,
    args,
    dataset
):        
    model.eval()
    device = model.llm.device
    benchmark_answers = inputs['answers']
    
    bsz = inputs["enc_doc_ids"].shape[0]
    pprint(bsz)
    pprint(inputs["enc_doc_ids"].shape[1])
    ORG_LEN = torch.sum(inputs["enc_doc_mask"][0]).item()

    for key, value in inputs.items():
        if key != "answers":
            inputs[key] = value.to(device)
    

    if args.split:
        segments, segments_mask = split_to_segments(
            inputs["enc_doc_ids"], 
            inputs["enc_doc_mask"], 
            args.segment_size
        )
    else:
        segments = [inputs["enc_doc_ids"]]
        segments_mask = [inputs["enc_doc_mask"]]
    
    pprint(len(segments))
    if args.post_append:
        aligned_memorys, memorys_mask = zip(*[
            model.generate_post_append_memorys(
                input_ids=s, 
                input_mask=m,
                merge_size=args.merge_size
            )
            for s, m in zip(segments, segments_mask)
        ])  
    elif args.enable_encore:
        # pprint("I am in encore memorys")
        start_time = time.time()
        aligned_memorys, memorys_masks, _ = model.generate_ram_memorys(
            input_ids_list=segments,
            input_mask_list=segments_mask,
            query_ids=inputs['enc_que_ids'],
            query_input_mask=inputs['enc_que_mask'],
            accumulation_ratio=args.top_p,
            pos_seg_ids=None,
            compute_contrastive=True
        ) 
        pprint(aligned_memorys.shape)
        pprint(memorys_masks.shape)
        compression_time = time.time() - start_time
    elif args.launch_tkdr:
        aligned_memorys = None
        memorys_masks = None
        if args.autoregressive:
            start_time = time.time()
            for idx, (s, m) in enumerate(zip(segments, segments_mask)):
                # assert 0
                aligned_memorys, memorys_masks = model.generate_autoregressive_tkdr_memorys(
                    input_ids=s, 
                    input_mask=m,
                    merge_size=args.merge_size,
                    query_ids=inputs['enc_que_ids'],
                    query_input_mask=inputs['enc_que_mask'],
                    pre_mem_embeds=aligned_memorys,
                    pre_mem_attention_mask=memorys_masks,
                    end_flag=(idx == len(segments) - 1)
                ) 
            compression_time = time.time() - start_time
        else:
            start_time = time.time()
            aligned_memorys, memorys_masks = zip(*[
                model.generate_tkdr_memorys(
                    input_ids=s, 
                    input_mask=m,
                    merge_size=args.merge_size,
                    query_ids=inputs["enc_que_ids"],
                    query_input_mask=inputs["enc_que_mask"]
                )
                for s, m in zip(segments, segments_mask)
            ])  
            compression_time = time.time() - start_time
    else:
        if args.autoregressive:
            aligned_memorys = None
            memorys_mask = None
            for s, m in zip(segments, segments_mask):
                aligned_memorys, memorys_mask = model.generate_autoregressive_pooling_memorys(
                    input_ids=s, 
                    input_mask=m,
                    merge_size=args.merge_size,
                    pre_mem_embeds=aligned_memorys,
                    pre_mem_attention_mask=memorys_mask
                )      
        else:
            start_time = time.time()
            aligned_memorys, memorys_masks = zip(*[
                model.generate_pooling_memorys(
                    input_ids=s, 
                    input_mask=m,
                    merge_size=args.merge_size
                )
                for s, m in zip(segments, segments_mask)
            ])
            compression_time = time.time() - start_time
    
    if not args.enable_encore:
        if not args.autoregressive:
            if len(aligned_memorys) == 1:
                aligned_memorys = aligned_memorys[0]
                memorys_masks = memorys_masks[0]
            else:
                aligned_memorys = torch.cat(aligned_memorys, dim=1)
                memorys_masks = torch.cat(memorys_masks, dim=1)
    
    pprint(aligned_memorys.shape[1])
    # COMP_LEN = aligned_memorys.shape[1]
    COMP_LEN = torch.sum(memorys_masks[0]).item()
    que_embeds = model.llm.model.embed_tokens(inputs["llm_que_ids"])
    
    start_time = time.time()
    llm_input_embeds = torch.cat((
        aligned_memorys,
        que_embeds
    ), dim=1)
    
    llm_attention_mask = torch.cat((
        memorys_masks, \
        inputs["llm_que_mask"]
    ), dim=1)

    pprint(inputs["enc_doc_ids"].shape)
    pprint(inputs["enc_doc_mask"].shape)

    outputs = model.llm.generate(
        inputs_embeds=llm_input_embeds,
        attention_mask=llm_attention_mask,
        max_new_tokens=dataset2maxgen[dataset],
        # max_new_tokens=64,
        eos_token_id=model.llm_tokenizer.eos_token_id
    )  
    mem_info = get_full_pytorth_memory_mb()
    pprint(mem_info["gpu_mem_reserved_mb"])

    infer_time = time.time() - start_time
    llm_generations = [elem for elem in model.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)]
    question = model.llm_tokenizer.batch_decode(inputs['llm_que_ids'], skip_special_tokens=True)
    pprint(question)
    benchmark_function_acc = benchmark_function_map['accuracy']
    benchmark_function_em = benchmark_function_map['em']
    benchmark_function_f1 = benchmark_function_map['f1']
    pprint(llm_generations[0])
    pprint(benchmark_answers[0])
    # exit(0)
    score_values_acc = [benchmark_function_acc(generation, answer) for generation, answer in zip(llm_generations, benchmark_answers)]
    score_values_em = [benchmark_function_em(generation, answer) for generation, answer in zip(llm_generations, benchmark_answers)]
    score_values_f1 = [benchmark_function_f1(generation, answer) for generation, answer in zip(llm_generations, benchmark_answers)]
    
    scores_acc = torch.tensor(score_values_acc, device=device)
    scores_em = torch.tensor(score_values_em, device=device)
    scores_f1 = torch.tensor(score_values_f1, device=device)

    pprint(1.0 * ORG_LEN / COMP_LEN)
    time_dict = [
        {
            "compression_time": compression_time,
            "infer_time": infer_time,
            "generation": llm_generations,
            "answers": benchmark_answers,
            "latency": compression_time + infer_time,
            "memory": mem_info["gpu_mem_reserved_mb"],
            "org_len": ORG_LEN,
            "comp_len": COMP_LEN
        }
    ]
    return scores_acc, scores_em, scores_f1, time_dict

def benchmark(
        model, 
        dataloader, 
        training_args,
        dataset
    ):
    # benchmark_bar = tqdm(
    #     total=len(dataloader), leave=True, dynamic_ncols=True,
    #     disable=not accelerator.is_main_process, desc='benchmark'
    # )
    # error_cnt = 0 
    model.eval()
    total_score_acc = 0
    total_score_em = 0
    total_score_f1 = 0
    total_num = 0
    outputs_host = []

    for inputs in tqdm(dataloader):
        # try:
        scores_acc, scores_em, scores_f1, time_dict = benchmark_step(
                            model, 
                            inputs,
                            args=training_args,
                            dataset=dataset 
                        )
        # scores_host += (accelerator.gather_for_metrics(scores),)
        total_score_acc += torch.sum(scores_acc)
        total_score_em += torch.sum(scores_em)
        total_score_f1 += torch.sum(scores_f1)
        
        total_num += scores_acc.shape[0]

        current_score_acc = 1.0 * total_score_acc / total_num
        current_score_em = 1.0 * total_score_em / total_num
        current_score_f1 = 1.0 * total_score_f1 / total_num
        
        pprint(current_score_acc)
        pprint(current_score_em)
        pprint(current_score_f1)
        pprint(time_dict[0]["compression_time"])
        pprint(time_dict[0]["infer_time"])
        outputs_host += time_dict

    avg_score_acc = 1.0 * total_score_acc / total_num
    avg_score_em = 1.0 * total_score_em / total_num
    avg_score_f1 = 1.0 * total_score_f1 / total_num
    
    return avg_score_acc, avg_score_em, avg_score_f1, outputs_host

def load_data(
        args,
        cmp_tokenizer,
        llm_tokenizer,
        training_args   
    ):
    # print(args.data_path)
    # exit(0)
    dataset = InferDataset(
        filepath=args.data_path,
        cmp_tokenizer=cmp_tokenizer,
        llm_tokenizer=llm_tokenizer,
        max_doc_tokens=args.max_doc_tokens,
        max_num_documents=args.num_eval_documents,
        llm_with_neg_documents=True,
        instruction_name=args.instruction_name,
        prefix_type=training_args.prefix_type
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    return data_loader

def load_restatement_data(
        args,
        model,
        training_args   
    ):

    dataset = TrainDataset(
        filepath=os.path.join(args.data_path, f'PwC_test.jsonl'),
        model=model,
        cmp_tokenizer=model.tokenizer,
        llm_tokenizer=model.llm_tokenizer,
        max_doc_tokens=args.max_doc_tokens,
        que_mask_ratio=args.question_mask_ratio,
        max_num_documents=args.max_num_documents,
        min_num_documents=args.min_num_documents,
        random_num_documents=args.random_num_documents,
        num_gold_documents=args.num_gold_documents,
        use_answer_as_target=args.use_answer_as_target,
        instruction_name=args.instruction_name,
        gold_first_for_kd=args.gold_first_for_kd,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    return data_loader

dataset2batchsize = {
    'NQ': 1,
    'HQA': 1,
    'MQA': 1,
    'NARRA': 1,
    'NEWS': 1,
    'NYC': 1,
    'CA': 1,
    'TKY': 1
}    

def main():
    seed_everything(42)
    parser = transformers.HfArgumentParser((RAMArguments, ModelArguments, TrainingArguments))
    args, model_args, training_args = parser.parse_args_into_dataclasses()

    data_paths = [args.data_path]
    
    pprint(data_paths)

    lora_config = None
    model = LLMCompressor(
        model_args, 
        training_args, 
        lora_config, 
        args.max_doc_tokens
    )

    if model.tokenizer.pad_token is None:
        model.tokenizer.add_special_tokens({'pad_token': '<encoder_pad>'})
        model.llm_encoder.resize_token_embeddings(len(model.tokenizer))

    if model.llm_tokenizer.pad_token is None:
        model.llm_tokenizer.add_special_tokens({'pad_token': '<llm_pad>'})
        model.llm.resize_token_embeddings(len(model.llm_tokenizer))


    for only_data_path in data_paths:
        args.data_path = only_data_path
        dataset = args.data_path.split('-')[-1].split(".")[0]
        training_args.per_device_eval_batch_size = dataset2batchsize[dataset]
        
        if args.restatement:
            test_dataloader = load_restatement_data(
                args, 
                model,
                training_args
            )
        else:
            test_dataloader = load_data(
                args, 
                model.tokenizer,
                model.llm_tokenizer,
                training_args         
            )
        
        # exit(0)

        check_point_path = training_args.resume_from_checkpoint
        # logger.info(f'build model and load checkpoint from {check_point_path}')
        print(f'build model and load checkpoint from {check_point_path}')
        # for .bin model
        # state_dict = torch.load(check_point_path, map_location='cpu')
        if ".bin" in check_point_path:
            state_dict = torch.load(check_point_path)
        else:
            state_dict = load_file(check_point_path)
        
        model.load_state_dict(state_dict)
        model = model.to(torch.bfloat16)
        model = model.to(device)

        if "," in training_args.top_p_list:
            top_p_list = [float(e) for e in training_args.top_p_list.split(",")]
        else:
            top_p_list = [float(training_args.top_p_list)]

        for top_p in top_p_list:
            training_args.top_p = top_p
            avg_scores_acc, avg_scores_em, avg_scores_f1, benchmark_outputs = benchmark(
                model, 
                test_dataloader, 
                training_args=training_args,
                dataset=dataset
            )
            
            total_org_len = sum(el["org_len"] for el in benchmark_outputs)
            total_comp_len = sum(el["comp_len"] for el in benchmark_outputs)
            appro_c_r = round(1.0 * total_org_len / total_comp_len, 1)

            # Sequentially write C_R ACC EM F1
            with open(os.path.join(training_args.output_dir, f"{dataset}_args_{training_args.top_p}_{training_args.index}.txt"), "w") as f:
                f.write(str(appro_c_r) + '\n')
                f.write(str(avg_scores_acc.item()) + '\n')
                f.write(str(avg_scores_em.item()) + '\n')
                f.write(str(avg_scores_f1.item()) + '\n')
            
            with open(os.path.join(training_args.output_dir, f"{dataset}_args_{training_args.top_p}_{training_args.index}.json"), "w") as f:
                json.dump(benchmark_outputs, f, indent=4)

if __name__ == "__main__":
    main()

