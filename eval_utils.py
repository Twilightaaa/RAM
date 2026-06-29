import re
import string
from collections import Counter


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value):
        excluded = set(string.punctuation)
        return "".join(character for character in value if character not in excluded)

    return " ".join(remove_articles(remove_punctuation(str(text).lower())).split())


def compute_exact(reference, prediction):
    return int(normalize_answer(prediction) == normalize_answer(reference))


def compute_f1(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)
