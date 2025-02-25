import datasets 

local_dir = "data/"
dataset = datasets.load_from_disk(local_dir)
print(dataset)