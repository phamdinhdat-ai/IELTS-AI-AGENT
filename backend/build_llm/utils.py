from datasets import load_dataset


local_dir = "data/"
ds = load_dataset("chillies/IELTS_essay_human_feedback")
ds.save_to_disk(local_dir)
