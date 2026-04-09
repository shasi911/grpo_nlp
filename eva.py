import json

file_path = "outputs/baseline_results.jsonl"

cat1 = 0  # format=1, answer=1
cat2 = 0  # format=1, answer=0
cat3 = 0  # format=0, answer=0

examples_cat1 = []
examples_cat2 = []
examples_cat3 = []

with open(file_path, "r") as f:
    for line in f:
        data = json.loads(line)
        fmt = data["format_reward"]
        ans = data["answer_reward"]

        if fmt == 1 and ans == 1:
            cat1 += 1
            if len(examples_cat1) < 3:
                examples_cat1.append(data)

        elif fmt == 1 and ans == 0:
            cat2 += 1
            if len(examples_cat2) < 3:
                examples_cat2.append(data)

        elif fmt == 0:
            cat3 += 1
            if len(examples_cat3) < 3:
                examples_cat3.append(data)

total = cat1 + cat2 + cat3

print("Category 1 (correct format + correct answer):", cat1)
print("Category 2 (correct format, wrong answer):", cat2)
print("Category 3 (wrong format):", cat3)

print("Accuracy:", cat1 / total)