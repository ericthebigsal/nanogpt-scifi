# nanoGPT/config/train_scifi_char.py
# Config for training a small char-level GPT on the sci-fi corpus, tuned for a laptop-scale
# Apple Silicon GPU (mps) run instead of nanoGPT's default multi-GPU/CUDA assumptions.
out_dir = 'out-scifi-char'
eval_interval = 250
eval_iters = 100
log_interval = 10

always_save_checkpoint = True

wandb_log = False

dataset = 'scifi_char'
# batch_size=16 x gradient_accumulation_steps=4 = the same 16,384 tokens/iteration as a
# single batch_size=64 step would give, on this 8GB dev machine, without touching model
# capacity: this is a memory-footprint tradeoff, not a change to the model being trained.
gradient_accumulation_steps = 4
batch_size = 16
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

device = 'mps'
compile = False
