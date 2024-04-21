from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import re
import os
import json
from argparse import ArgumentParser


def load_model(model_name):
    llm = LLM(model_name, seed=42)
    return llm


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

    *demonstrations, question = original_prompt.split("\n\n")
    demonstrations = demonstrations[:2]
    compressed_prompt = "\n\n".join(demonstrations) + "\n\n" + question

    return compressed_prompt


def compress_prompt_1(original_prompt, tokenizer=None):
    """
    Compress the given prompt to
    """

    *demonstrations, question = original_prompt.split("\n\n")
    demonstrations = demonstrations.sort()[:1]
    compressed_prompt = "\n\n".join(demonstrations) + "\n\n" + question

    return compressed_prompt


def compress_prompt_2(original_prompt, tokenizer=None):
    """
    Compress the given prompt to
    """

    // 省略解答过程, 只保留结果

    *demonstrations, question = original_prompt.split("\n\n")
    demonstrations = demonstrations.sort()[:1]
    compressed_prompt = "\n\n".join(demonstrations) + "\n\n" + question

    return compressed_prompt


compress_methods = [compress_prompt_0, compress_prompt_1, compress_prompt_2]


def generate_answer(prompts, model):
    """Generate answer for each text in texts"""

    sampling_params = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=150)

    answers = []
    for input_text in tqdm(prompts):
        output = model.generate(
            input_text, use_tqdm=False, sampling_params=sampling_params
        )
        answer = (
            output[0].outputs[0].text.split("\n\n")[0]
        )  # only keep the first paragraph
        answers.append(answer)

    return answers


def evaluate_answers(answers, labels):
    """Evaluate the answers"""
    scores = []
    for answer, label in zip(answers, labels):
        numbers = re.findall(r"\d+", answer)
        last_two_numbers = numbers[-2:]
        scores.append(any([label == number for number in last_two_numbers]))

    print("Accuracy: ", sum(scores) / len(scores))

    return scores


def test(prompts, labels, args, compress, case):
    print("case", case)
    compressed_prompts = []
    for p in prompts:
        compressed_prompt = compress(p)
        compressed_prompts.append(compressed_prompt)
        original_length += len(p)
        compressed_length += len(compressed_length)
    print("compress ratio: ", compressed_length / original_length)

    """Conduct Inference"""
    answers = generate_answer(compressed_prompts, model)

    """Evaluate the answers"""
    scores = evaluate_answers(answers, labels)

    """Save the results"""
    output_path = f"results/{args.model_name}_{args.demo_type}_{case}.json"
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, "w") as f:
        json.dump(
            [
                {"prompt": p, "label": l, "answer": a, "score": s}
                for p, l, a, s in zip(prompts, labels, answers, scores)
            ],
            f,
            indent=4,
        )


if __name__ == "__main__":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    """Args"""
    args = ArgumentParser()
    args.add_argument("--model_name", type=str, default="facebook/opt-1.3b")
    args.add_argument("--demo_type", type=str, default="all")
    args = args.parse_args()

    """Load everything we need"""
    model = load_model(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prompts, labels = load_dataset(args.demo_type)

    # """Compress the prompt"""
    original_length = 0
    compressed_length = 0

    for index, value in enumerate(compress_methods):
        test(prompts, labels, args, value, index)
