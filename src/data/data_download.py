
from datasets import load_dataset

# Method 1: Load dataset directly (most common)
# This downloads and caches the dataset automatically
dataset = load_dataset("openbmb/RLAIF-V-Dataset")  # Example: Stanford Question Answering Dataset

# Access different splits
train_data = dataset['train']
# validation_data = dataset['validation']

print(f"Training examples: {len(train_data)}")
# print(f"Validation examples: {len(validation_data)}")

# View first example
print("\nFirst example:")
print(train_data[0])

# # Method 2: Load specific configuration/subset
# dataset = load_dataset("glue", "mrpc")  # GLUE benchmark, MRPC task

# # Method 3: Load and save to disk
# dataset = load_dataset("imdb")
# dataset.save_to_disk("./imdb_dataset")  # Save locally

# # Method 4: Load from disk later
# from datasets import load_from_disk
# loaded_dataset = load_from_disk("./imdb_dataset")

# # Method 5: Stream large datasets (doesn't download everything at once)
# dataset = load_dataset("oscar", "unshuffled_deduplicated_en", streaming=True)
# for example in dataset['train'].take(5):
#     print(example)
    
# Installation required: pip install datasets