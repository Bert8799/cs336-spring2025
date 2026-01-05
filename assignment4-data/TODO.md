# TODO

## Part 4: Leaderboard — Filter Data for Language Modeling

### Assignment Goal

The goal of this assignment is to optimize the training data to minimize validation loss, rather than improving performance by modifying the model architecture or optimization procedure.  
Therefore, the training configuration and training script should not be modified, except for the dataset path and the Weights & Biases (wandb) attributes specified in the instructions.

### Planned Approach

- Download the Common Crawl WET files (`CC*.warc.wet.gz`, 5,000 files required) and the validation set `tokenized_paloma_c4_100_domains_validation.bin`.
- Write a script to filter high-quality language modeling data from a collection of Common Crawl WET files.
- Use the GPT-2 tokenizer via the `transformers` library to encode the filtered text into sequences of integer token IDs.
- Write a script to tokenize and serialize the filtered dataset into the required binary format for training.

### Deliverable

- The best validation loss achieved
- The corresponding learning curve
- A brief description of the data filtering and preprocessing steps used
