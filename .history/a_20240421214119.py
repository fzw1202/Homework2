import re
import os
import json
import re
from argparse import ArgumentParser


def load_dataset(demo_type):
    with open("data.jsonl", "r") as f:
        dataset = [json.loads(line) for line in f]

    assert (
        demo_type == "all"
        or demo_type == "complex"
        or demo_type == "easy"
        or demo_type == "mid"
    )
    if demo_type != "all":
        dataset = [d for d in dataset if d["demo_type"] == demo_type]

    return [d["prompt"] for d in dataset], [d["label"] for d in dataset]


def compress_prompt_0(original_prompt, tokenizer=None):
    """
    Compress the given prompt to
    """

    # 保留前两个示例

    *demonstrations, question = original_prompt.split("\n\n")
    demonstrations = demonstrations[:2]
    compressed_prompt = "\n\n".join(demonstrations) + "\n\n" + question

    return compressed_prompt


def compress_prompt_1(original_prompt, tokenizer=None):
    """
    Compress the given prompt to
    """

    # 只保留长度最短的两个示例

    *demonstrations, question = original_prompt.split("\n\n")
    demonstrations = sorted(demonstrations, key=len)[:1]
    compressed_prompt = "\n\n".join(demonstrations) + "\n\n" + question

    return compressed_prompt


def compress_prompt_2(original_prompt, tokenizer=None):
    """
    Compress the given prompt to
    """

    # 省略解答过程, 只保留结果

    *demonstrations, question = original_prompt.split("\n\n")
    for index, value in enumerate(demonstrations):
        sentences = value.split("\n")
        demonstrations[index] = sentences[0] + "\n" + sentences[-1]
    compressed_prompt = "\n\n".join(demonstrations) + "\n\n" + question

    return compressed_prompt


def compress_prompt_3(original_prompt, tokenizer=None):
    """
    Compress the given prompt to
    """

    # 取代人名为 A (前面有 Miss/Mr./Mrs.)

    *demonstrations, question = original_prompt.split("\n\n")
    name_pattern = r"(?:Miss\.|Mr\.|Mrs\.)\s+(\w+)"
    demonstrations = [
        re.sub(name_pattern, r"A", i) for i in demonstrations
    ] 
    compressed_prompt = "\n\n".join(demonstrations) + "\n\n" + question

    return compressed_prompt


def compress_prompt_4(original_prompt, tokenizer=None):
    """
    Compress the given prompt to
    """

    # 删除特定的单词和句子

    sentences = [
        "Let's think step by step",
    ]
    words = [
        "first",
        "Second",
        "Third",
        "Then",
        "And",
        "Therefore",
        "Thus",
        "Similarly",
        "That",
    ]

    *demonstrations, question = original_prompt.split("\n\n")
    for index, value in enumerate(demonstrations):
        s = value.split("\n")
        value = [i for i in s if i not in sentences]
        demonstrations[index] = "\n".join(value) + "\n"

    words_pattern = r"\b(?:" + "|".join(words) + r")(?:,)?\b"
    demonstrations = [
        re.sub(words_pattern, "", i) for i in demonstrations
    ]   
    compressed_prompt = "\n\n".join(demonstrations) + "\n\n" + question

    return compressed_prompt


def compress_prompt_all(original_prompt, tokenizer=None):
    compressed_prompt = compress_prompt_4(original_prompt)
    compressed_prompt = compress_prompt_3(compressed_prompt)

    return compressed_prompt


compress_methods = [
    compress_prompt_0,
    compress_prompt_1,
    compress_prompt_2,
    compress_prompt_3,
    compress_prompt_4,
    compress_prompt_all,
]


def evaluate_answers(answers, labels):
    """Evaluate the answers"""
    scores = []
    for answer, label in zip(answers, labels):
        numbers = re.findall(r"\d+", answer)
        scores.append(any([label == number for number in numbers]))

    print("Accuracy: ", sum(scores) / len(scores))

    return scores


def test_prompt(prompts, labels, args, compress, case):
    print("case: ", case)
    p = prompts[0]
    ol = len(p)

    print("------ original prompt ------\n", p)

    compressed_prompt = compress(p)
    cl = len(compressed_prompt)
    print("------ compressed prompt ------\n", compressed_prompt)
    print(ol, cl, "ratio: ", cl / ol)


if __name__ == "__main__":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    """Args"""
    args = ArgumentParser()
    args.add_argument("--model_name", type=str, default="facebook/opt-1.3b")
    args.add_argument("--demo_type", type=str, default="all")
    args = args.parse_args()

    """Load everything we need"""
    prompts, labels = load_dataset(args.demo_type)

    # """Compress the prompt"""
    original_length = 0
    compressed_length = 0

    for index, value in enumerate(compress_methods):
        test_prompt(prompts, labels, args, value, index)
