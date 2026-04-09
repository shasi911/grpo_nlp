import json
from typing import List, Callable
from vllm import LLM, SamplingParams

# IMPORTANT: adjust import based on your repo structure
from drgrpo_grader import r1_zero_reward_fn


def load_r1_zero_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r") as f:
        return f.read()


def load_math_validation(file_path: str):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            data.append({
                "question": item["problem"],   # confirm key if needed
                "answer": item["solution"]
            })
    return data


def format_prompts(data, template: str) -> List[str]:
    prompts = []
    for item in data:
        prompt = template.format(question=item["question"])
        prompts.append(prompt)
    return prompts


def evaluate_vllm(
    llm: LLM,
    prompts: List[str],
    answers: List[str],
    reward_fn: Callable,
    output_file: str
):
    
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    outputs = llm.generate(prompts, sampling_params)

    results = []

    correct = 0
    format_correct = 0

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        prompt = prompts[i]
        gold_answer = answers[i]

        reward_dict = reward_fn(generated_text, gold_answer)

        # expected keys: "format_reward", "answer_reward"
        fmt = reward_dict.get("format_reward", 0)
        ans = reward_dict.get("answer_reward", 0)

        format_correct += fmt
        correct += ans

        results.append({
            "prompt": prompt,
            "generation": generated_text,
            "gold_answer": gold_answer,
            "format_reward": fmt,
            "answer_reward": ans
        })

    # write results
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # print metrics
    total = len(results)
    print(f"Total: {total}")
    print(f"Format correct: {format_correct / total:.4f}")
    print(f"Answer correct: {correct / total:.4f}")


def main():
    # paths (adjust if needed)
    prompt_path = "/scratch/sfr5846/cse_587/assignment2-alignment/assignment2-alignment/alignment/prompts/r1_zero.prompt"
    data_path = "/scratch/sfr5846/cse_587/assignment2-alignment/assignment2-alignment/data/MATH/validation.jsonl"
    output_path = "/scratch/sfr5846/cse_587/assignment2-alignment/assignment2-alignment/outputs/baseline_results.jsonl"

    # load prompt template
    template = load_r1_zero_prompt(prompt_path)

    # load dataset
    data = load_math_validation(data_path)

    # format prompts
    prompts = format_prompts(data, template)
    answers = [item["answer"] for item in data]

    # initialize model
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B",trust_remote_code=True)

    # evaluate
    evaluate_vllm(
        llm=llm,
        prompts=prompts,
        answers=answers,
        reward_fn=r1_zero_reward_fn,
        output_file=output_path,
        
    )


if __name__ == "__main__":
    main()