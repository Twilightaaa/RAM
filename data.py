import json
from tqdm import tqdm

def process_jsonl(path, out_path):
    with open(path, "r", encoding="utf-8") as f:
        with open(out_path, "w", encoding="utf-8") as fout:
            for line in tqdm(f, desc="Processing"):
                line = line.strip()
                if not line:
                    continue
                json_obj = json.loads(line)

                question = json_obj["prompt"]
                ctxs = [{"isgold": False, "text": json_obj["input"], "title": ""}]
                answers = json_obj["answer"]

                item = {
                    "question": question,
                    "ctxs": ctxs,
                    "answers": answers
                }

                item_str = json.dumps(item, ensure_ascii=False)
                fout.write(item_str + "\n")

path = ""
out_path = ""
process_jsonl(path, out_path)