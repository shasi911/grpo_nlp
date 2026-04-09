import torch

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    full_ids_list = []
    masks_list = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        output_ids = tokenizer(output, add_special_tokens=False)["input_ids"]

        full_ids = prompt_ids + output_ids

        # mask: 0 for prompt, 1 for output (aligned with input_ids = full_ids[:-1])
        response_mask = [0]*(len(prompt_ids)-1) + [1]*len(output_ids)

        full_ids_list.append(full_ids)
        masks_list.append(response_mask)

    # Pad full_ids to max length, then slice to get input_ids and labels
    max_full_len = max(len(x) for x in full_ids_list)

    def pad_full(seq):
        return seq + [tokenizer.pad_token_id] * (max_full_len - len(seq))

    def pad_mask(seq):
        return seq + [0] * (max_full_len - 1 - len(seq))

    padded_full = [pad_full(x) for x in full_ids_list]

    input_ids = torch.tensor([x[:-1] for x in padded_full])
    labels = torch.tensor([x[1:] for x in padded_full])
    response_mask = torch.tensor([pad_mask(x) for x in masks_list], dtype=torch.bool)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }
