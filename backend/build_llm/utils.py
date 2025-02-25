from datasets import load_dataset
from  prompts import WRITTING_TASK2_PROMPT

local_dir = "data/"
ds = load_dataset("chillies/IELTS_essay_human_feedback")
ds.save_to_disk(local_dir)


EOS_TOKEN = '</s>'
def formatting_prompts_func(examples):
    essay_prompts = examples["prompt"]
    essays = examples["essay"]
    chosen = list(examples["chosen"])
    rejected = list(examples["rejected"])

    texts = []
    for p, e in zip(essay_prompts, essays):
        text = WRITTING_TASK2_PROMPT.format(p, e, "") + '</s>'
        texts.append(text)

    chosen_evals = []
    rejected_evals = []
    for c, r in zip(chosen, rejected):
        chosen_evals.append(c + '</s>')
        rejected_evals.append(r + '</s>')

    return {"prompt": texts, "chosen": chosen_evals, "rejected": rejected_evals}

