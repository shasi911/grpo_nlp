import json

# your template
template = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer.
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags.

User: {question}
Assistant: <think>"""

# input and output files
path = "/scratch/sfr5846/cse_587/assignment2-alignment/assignment2-alignment/data/MATH/"
input_file = "validation.jsonl"
output_file = "prompts.txt"

with open(path+input_file, "r") as fin, open(path+output_file, "w") as fout:
    for line in fin:
        data = json.loads(line)  

        question = data["problem"]   

        # replace {question} in template
        prompt = template.format(question=question)

        # write to file
        fout.write(prompt + "\n\n")   # spacing between prompts