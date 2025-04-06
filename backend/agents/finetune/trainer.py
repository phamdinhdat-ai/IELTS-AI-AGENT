import datasets 
from backend.agents.utils.utils import formatting_prompts_func
from unsloth import FastLanguageModel
import torch
from unsloth import PatchDPOTrainer
from transformers import TrainingArguments
from trl import DPOTrainer
from backend.agents.prompts.training_prompts import WRITTING_TASK2_PROMPT

PatchDPOTrainer()

max_seq_length = 2048
dtype = None
load_in_4bit = True


local_dir = "data/"
dataset = datasets.load_from_disk(local_dir)
print(dataset)
dpo_dataset = dataset.map(formatting_prompts_func, batched=True).remove_columns(['essay'])


MODEL_NAME = "dat-ai/IELTS_Writing_task2"
model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,# "IELTS_evaluation_lora_model",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
)

args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    num_train_epochs=1,
    learning_rate=1e-5,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.0,
    lr_scheduler_type="linear",
    seed=42,
    output_dir="dpo_outputs")


dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=args,
    beta=0.1,
    tokenizer=tokenizer,
    train_dataset=dpo_dataset['test'],
    max_length=1024,
    max_prompt_length=1024
)

dpo_trainer.train()
model.push_to_hub("dat-ai/DPO_IELTS_Writing_task2", token="") # Online saving

prompt = """
In most countries, animal and plant species are declining rapidly. What are the causes of this? What measures could be done to prevent this decline?
"""
essay = """
The species of animals and plants are rapidly decreasing in most countries. In this essay, I will examine the factors that contribute towards declining animal and plant species and propose some solutions to that.

There are a few factors that contribute to the falling species of animals and plants. Firstly, the main cause for this issue probably is human activities. There are some people who like to kill animals and plants for their own purpose such as for collection. Secondly, the other reason why is this happening is because of nature degradation. Nowadays, climate change is getting worse in most countries  leading to natural selection for animals and plants. Therefore, their species witnessed a rapid fallen.

A few ways can be taken to prevent the decline of animal and plant species. Firstly, government plays a crucial role. They have to be aware of this issue and then  formulate some regulations to prevent animal and plant hunting. Although the law is already generated, the government should be more active to make sure the regulation is well-running. Furthermore, as citizens, we can drive a movement to raise awareness about this issue. Maybe, some people will underestimate what can citizens do with this little, but I believe if we hold hands together, we can make a change.

In conclusion, the species of animals and plants are declining rapidly  mainly caused by human activities that do illegal hunting. This essay suggested that the ways to prevent this problem are twofold: to generate strict regulation and to create a movement in order to raise society's awareness.
"""

FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    WRITTING_TASK2_PROMPT.format(
        prompt,
        essay,
        "",
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
output = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 2048)
# print(output)