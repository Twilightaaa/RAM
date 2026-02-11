import torch
import random
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from icecream import ic as pprint
import json

instructions_map = {
    'base': 'Write a high-quality answer for the given question using only the provided search results(some of which might be irrelevant).\n\n',
    'short': 'Answer the Question based on the given text. Only give me the answer and do not output any other words.\nText:',
    'repeat': 'Restate the aforementioned Text.',
    'ft_prefix': 'Search results:',
    'rs_prefix': 'Text:',
    'pwc': 'Answer the Question based on the given Text. \nText:',
    'empty': ''
}

def format_document(document, tokenizer, max_tokens=None):
    if max_tokens is not None:     
        return tokenizer.decode(
                tokenizer(
                document['title'] + ' ' + document['text'] if 'title' in document else document['text'],
                add_special_tokens=False,
            )['input_ids'][:max_tokens]
        )
    
    return tokenizer.decode(
            tokenizer(
            document['title'] + ' ' + document['text'] if 'title' in document else document['text'],
            add_special_tokens=False,
        )['input_ids']
    )

def trunc_text(text, tokenizer, max_tokens=None):
    return tokenizer.decode(
            tokenizer(
            text,
            add_special_tokens=False,
        )['input_ids'][:max_tokens]
    )

class TrainDataset(Dataset):
    def __init__(
        self,
        filepath,
        model,
        max_doc_tokens,
        instruction_name,
        max_num_documents=None,
        min_num_documents=None,
        random_num_documents=False,
        **kwargs,
    ):
        self.dataset = load_dataset('json', data_files=filepath, split='train')
        self.max_doc_tokens = max_doc_tokens
        self.model = model
        self.cmp_tokenizer = model.tokenizer
        self.llm_tokenizer = model.llm_tokenizer
        self.max_num_documents = max_num_documents
        self.min_num_documents = min_num_documents
        self.random_num_documents = random_num_documents
        self.prefix_type = kwargs['prefix_type']

        self.instruction_text = instructions_map[instruction_name]
        self.prefix_text = instructions_map[self.prefix_type]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        example = self.dataset[index]
        question = example['question']
        documents = [
            format_document(document, self.cmp_tokenizer)
            for document in example['ctxs'] 
        ]

        return {
            'question': question,
            'documents': documents,
        }

    def collate_fn(self, batch):
        if len(batch) == 0:
            return {}
        
        num_documents = (
            random.randint(self.min_num_documents, self.max_num_documents)
            if self.random_num_documents else self.max_num_documents
        )
        
        enc_documents = [trunc_text("\n".join(instance["documents"][:num_documents]), self.cmp_tokenizer, self.max_doc_tokens)\
            for instance in batch]
        enc_questions = [instance['question'] for instance in batch]
        llm_questions = ['\nQuestion:' + instance['question'] + '\nAnswer:' for instance in batch]
        llm_targets = [trunc_text("\n".join(instance["documents"][:num_documents]), self.cmp_tokenizer, self.max_doc_tokens)\
            for instance in batch]
        llm_instructions = [self.instruction_text for _ in batch]
        llm_prefix_tokens = [self.prefix_text for _ in enc_questions]

        repeat_tokens = ['Restate the aforementioned Text.' for _ in enc_questions]
        enc_prefix_outputs = self.llm_tokenizer(llm_prefix_tokens, return_tensors='pt', padding=True)
        enc_repeat_outputs = self.llm_tokenizer(repeat_tokens, return_tensors='pt', padding=True, add_special_tokens=False)
        enc_que_outputs = self.cmp_tokenizer(enc_questions, return_tensors='pt', padding=True, add_special_tokens=False)
        enc_doc_outputs = self.cmp_tokenizer(enc_documents, return_tensors='pt', padding=True, add_special_tokens=False)

        total_mem_size = self.model.mem_size * self.model.compute_num_segments(len(enc_doc_outputs.input_ids[0]))
        llm_ins_outputs = self.llm_tokenizer(llm_instructions, return_tensors='pt', padding=True)
        llm_que_outputs = self.llm_tokenizer(llm_questions, return_tensors='pt', padding=True, add_special_tokens=False)
        memorys = torch.zeros((llm_ins_outputs["input_ids"].shape[0], total_mem_size))

        def right_padding(value, padding_value):
            padded_value = pad_sequence(
                [torch.tensor(v) for v in value],
                batch_first=True,
                padding_value=padding_value,
            )
            return padded_value

        llm_tgt_outputs = [self.llm_tokenizer(ans, add_special_tokens=False).input_ids for ans in llm_targets]
        llm_tgt_tokens = right_padding(llm_tgt_outputs, self.llm_tokenizer.pad_token_id)
        llm_tgt_mask = right_padding([[1] * len(elem) for elem in llm_tgt_outputs], 0)

        llm_ins_mask = llm_ins_outputs.attention_mask
        llm_que_mask = llm_que_outputs.attention_mask

        memorys_mask = torch.ones((memorys.shape[0], memorys.shape[1]))
        llm_attention_mask = torch.cat((llm_ins_mask, memorys_mask, llm_que_mask, llm_tgt_mask), dim=1)
        llm_input_ids = torch.cat((llm_ins_outputs["input_ids"], memorys, \
                                   llm_que_outputs["input_ids"], llm_tgt_tokens), dim=1)

        llm_labels = torch.full_like(llm_attention_mask, -100)
        llm_labels[:, -llm_tgt_tokens.size(1):] = llm_tgt_tokens.masked_fill(
            ~llm_tgt_mask.bool(), -100,
        )

        return {
            'enc_doc_ids': enc_doc_outputs.input_ids,
            'enc_doc_mask': enc_doc_outputs.attention_mask,
            'enc_que_ids': enc_que_outputs.input_ids,
            'enc_que_mask': enc_que_outputs.attention_mask,
            'llm_input_ids': llm_input_ids,
            'llm_input_mask': llm_attention_mask,
            'llm_ins_ids': llm_ins_outputs.input_ids,
            'llm_ins_mask': llm_ins_mask,
            'labels': llm_labels,
            'enc_prefix_ids': enc_prefix_outputs.input_ids,
            'enc_prefix_mask': enc_prefix_outputs.attention_mask,
            'enc_repeat_ids': enc_repeat_outputs.input_ids,
            'enc_repeat_mask': enc_repeat_outputs.attention_mask
        }
    

class InferDataset(Dataset):
    def __init__(
        self,
        filepath,
        cmp_tokenizer,
        llm_tokenizer,
        max_doc_tokens,
        instruction_name,
        max_num_documents=None,
        **kwargs,
    ):
        self.dataset = load_dataset('json', data_files=filepath, split='train')
        self.max_doc_tokens = max_doc_tokens
        self.cmp_tokenizer = cmp_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.max_num_documents = max_num_documents
        self.prefix_type = kwargs['prefix_type']

        self.cmp_tokenizer.padding_side = 'left'
        self.llm_tokenizer.padding_side = 'left'
            
        if self.cmp_tokenizer.pad_token is None:
            self.cmp_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.cmp_tokenizer.pad_token = "[PAD]"
            self.cmp_tokenizer.pad_token_id = self.cmp_tokenizer.convert_tokens_to_ids("[PAD]")
            
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.llm_tokenizer.pad_token = "[PAD]"
            self.llm_tokenizer.pad_token_id = self.cmp_tokenizer.convert_tokens_to_ids("[PAD]")

        self.instruction_text = instructions_map[instruction_name]
        self.prefix_text = instructions_map[self.prefix_type]

    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):
        example = self.dataset[index]
        question = example['question']
        
        documents = [
            format_document(document, self.cmp_tokenizer, None)
            for document in example['ctxs']
        ]

        answers = example['answers']

        return {
            'question': question,
            'documents': documents,
            'answers': answers,
        }


    def collate_fn(self, batch):
        if len(batch) == 0:
            return {}

        enc_documents = []
        # llm_prefix_tokens = []
        for instance in batch:
            instance_enc_candidate_documents = instance['documents']
            instance_enc_documents = '\n'.join(instance_enc_candidate_documents)
            enc_documents.append(instance_enc_documents)

        enc_questions = [instance['question'] for instance in batch]
        llm_questions = ['Question:' + instance['question'] + '\nAnswer:' for instance in batch]
        llm_instructions = [self.instruction_text for _ in batch]
        answers = [instance['answers'] for instance in batch]

        llm_prefix_tokens = [self.prefix_text for _ in enc_questions]
        repeat_tokns = ['Restate the aforementioned Text.' for _ in enc_questions]
        enc_prefix_outputs = self.llm_tokenizer(llm_prefix_tokens, return_tensors='pt', padding=True, add_special_tokens=False)
        enc_repeat_outputs = self.llm_tokenizer(repeat_tokns, return_tensors='pt', padding=True, add_special_tokens=False)

        enc_que_outputs = self.cmp_tokenizer(enc_questions, return_tensors='pt', padding=True, add_special_tokens=False)
        enc_doc_outputs = self.cmp_tokenizer(enc_documents, return_tensors='pt', padding=True, add_special_tokens=False)
        llm_ins_outputs = self.llm_tokenizer(llm_instructions, return_tensors='pt', padding=True)
        llm_que_outputs = self.llm_tokenizer(llm_questions, return_tensors='pt', padding=True, add_special_tokens=False)

        return {
            'enc_doc_ids': enc_doc_outputs.input_ids,
            'enc_que_ids': enc_que_outputs.input_ids,
            'enc_doc_mask': enc_doc_outputs.attention_mask,
            'enc_que_mask': enc_que_outputs.attention_mask,
            'llm_ins_ids': llm_ins_outputs.input_ids,
            'llm_que_ids': llm_que_outputs.input_ids,
            'llm_ins_mask': llm_ins_outputs.attention_mask,
            'llm_que_mask': llm_que_outputs.attention_mask,
            'answers': answers,
            'enc_prefix_ids': enc_prefix_outputs.input_ids,
            'enc_prefix_mask': enc_prefix_outputs.attention_mask,
            'enc_repeat_ids': enc_repeat_outputs.input_ids,
            'enc_repeat_mask': enc_repeat_outputs.attention_mask
        }