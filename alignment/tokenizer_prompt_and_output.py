import torch

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    input_ids_list = []
    labels_list = []
    masks_list = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        output_ids = tokenizer(output, add_special_tokens=False)["input_ids"]

        full_ids = prompt_ids + output_ids

        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # mask: 0 for prompt, 1 for output
        response_mask = [0]*(len(prompt_ids)-1) + [1]*len(output_ids)

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        masks_list.append(response_mask)

    # padding
    max_len = max(len(x) for x in input_ids_list)

    def pad(seq, pad_value):
        return seq + [pad_value]*(max_len - len(seq))

    input_ids = torch.tensor([pad(x, tokenizer.pad_token_id) for x in input_ids_list])
    labels = torch.tensor([pad(x, -100) for x in labels_list])
    response_mask = torch.tensor([pad(x, 0) for x in masks_list])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }