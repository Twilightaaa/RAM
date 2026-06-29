import math

# DEFAULT_NQ_PATH = "/hpc2hdd/home/zwu791/hzj/ali-show/data-nq/nq_ram.jsonl"
DEFAULT_NQ_PATH = "/path/to/nq_ram.jsonl"

def _error(sample_index, message):
    location = f"sample {sample_index}" if sample_index is not None else "sample"
    raise ValueError(f"Invalid NQ {location}: {message}")


def format_document(document, sample_index=None):
    if not isinstance(document, dict):
        _error(sample_index, "ctxs entries must be objects")
    text = document.get("text")
    if not isinstance(text, str) or not text.strip():
        _error(sample_index, "ctxs[*].text must be a non-empty string")
    title = document.get("title", "")
    if title is None:
        title = ""
    if not isinstance(title, str):
        _error(sample_index, "ctxs[*].title must be a string")
    return f"{title.strip()} {text.strip()}".strip()


def prepare_nq_example(example, sample_index=None):
    for field in ("question", "ctxs", "answers"):
        if field not in example:
            _error(sample_index, f"missing field '{field}'")

    question = example["question"]
    ctxs = example["ctxs"]
    answers = example["answers"]
    if not isinstance(question, str) or not question.strip():
        _error(sample_index, "question must be a non-empty string")
    if not isinstance(ctxs, list) or not ctxs:
        _error(sample_index, "ctxs must be a non-empty list")
    if not isinstance(answers, list) or not answers:
        _error(sample_index, "answers must be a non-empty list")

    normalized_answers = [str(answer) for answer in answers]
    if not normalized_answers[0].strip():
        _error(sample_index, "answers[0] must be non-empty")

    return {
        "context": "\n".join(format_document(document, sample_index) for document in ctxs),
        "question": question,
        "decoder_prompt": f"Question:{question}\nAnswer:",
        "answer": normalized_answers[0],
        "answers": normalized_answers,
    }


def _find_subsequence_spans(sequence, pattern):
    if not pattern or len(pattern) > len(sequence):
        return []
    width = len(pattern)
    return [
        (start, start + width)
        for start in range(len(sequence) - width + 1)
        if sequence[start : start + width] == pattern
    ]


def build_positive_segment_mask(context_ids, answers, tokenizer, segment_size):
    if segment_size <= 0:
        raise ValueError("segment_size must be positive")

    mask = [False] * math.ceil(len(context_ids) / segment_size)
    for answer in answers:
        answer = str(answer).strip()
        if not answer:
            continue
        for variant in {answer, f" {answer}"}:
            answer_ids = tokenizer.encode(variant, add_special_tokens=False)
            for start, end in _find_subsequence_spans(context_ids, answer_ids):
                first_segment = start // segment_size
                last_segment = (end - 1) // segment_size
                for segment_index in range(first_segment, last_segment + 1):
                    mask[segment_index] = True
    return mask
