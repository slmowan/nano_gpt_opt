import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import inspect
import numpy as np
from typing import Callable, Iterable, Tuple
import matplotlib.pyplot as plt
from loguru import logger
import os
import json
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset, get_worker_info

# custom optimizers
from rmsprop_optimizer import RMSProp
from sgd_momentum_optimizer import SGDMomentum


# Model architecture: nanogpt
class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


# Adam Optimizer (baseline): https://optimai-lab.github.io/LLM-OPT/lectures/Chapter3/LLM.html#llm-training
class Adam(torch.optim.Optimizer):
    """Implements Adam algorithm with weight decay fix."""
    
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0

                # State initialization
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # Adam update
                u = exp_avg / denom
                p.add_(u, alpha=-step_size)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


# Dataset class
class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, device_batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.device_batch_size = device_batch_size
        self.max_length = max_length

    def __iter__(self):
        iter_data = iter(self.data)
        batch = []
        for example in iter_data:
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch.append(tokenized_example)
            if len(batch) == self.device_batch_size:
                yield self._format_batch(batch)
                batch = []
        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        return input_ids



# Evaluation
def collate_fn(batch_list):
    batch = torch.stack([torch.Tensor(example["input_ids"]).long() for example in batch_list])
    return batch


def batch_fn(dataset, batch_size):
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            batch = collate_fn(batch)
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def preprocess_batched(batch, tokenizer, max_length):
    batch = tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return batch


@torch.no_grad()
def evaluate_model(model, val_data, preprocess_batched, pad_idx, device, batch_size):
    val_data = val_data.shuffle(seed=42)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: batch_fn(
        val_data_mapped, batch_size
    )

    target_eval_tokens = 1_000_000 
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 0

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        input_ids = batch.to(device)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = pad_idx
        labels[labels == pad_idx] = -100
        labels = labels.to(device)

        _, loss = model(input_ids, targets=labels)
        total_loss += loss.detach()

        evaluated_on_tokens += (batch != pad_idx).sum().item()

    total_loss = total_loss / total_batches

    return total_loss, evaluated_on_tokens




# Training
def train_model(
    optimizer_name,
    optimizer,
    scheduler,
    model,
    dataloader,
    val_data,
    tokenizer,
    device,
    config
):
    logger.info(f"Starting training with {optimizer_name}")
    
    # Unpack config
    num_training_steps = config['num_training_steps']
    gradient_accumulation = config['gradient_accumulation']
    grad_clipping = config['grad_clipping']
    print_freq = config['print_freq']
    eval_every = config['eval_every']
    pad_idx = config['pad_idx']
    device_batch_size = config['device_batch_size']
    max_length = config['max_length']
    
    # Initialize tracking
    global_step = 0
    update_step = 0
    tokens_seen = 0
    
    train_losses = []
    eval_losses = []
    eval_perplexities = []
    
    # Start of training loop
    for batch_idx, batch in enumerate(dataloader):
        global_step += 1
        
        if update_step > num_training_steps:
            logger.info(f"Reached max number of update steps ({num_training_steps}). Stopping training.")
            break
        
        input_ids = batch.to(device)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = pad_idx
        labels[labels == pad_idx] = -100
        labels = labels.to(device)
        tokens_seen += (input_ids != pad_idx).sum().item()
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16): # automatic mixed precision
            logits, loss = model(input_ids, targets=labels)
        
        scaled_loss = loss / gradient_accumulation
        scaled_loss.backward()
        
        if global_step % gradient_accumulation != 0:
            continue
        
        train_losses.append((loss.item(), update_step))
        
        if update_step % print_freq == 0:
            logger.info(f"Update step: {update_step}/{num_training_steps} | loss: {loss.item():.4f}")
        
        if grad_clipping != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        update_step += 1
        
        if eval_every > 0 and ((update_step % eval_every == 0) or (update_step == num_training_steps)):
            model.eval()

            total_loss, evaluated_on_tokens = evaluate_model(
                model, 
                val_data,
                lambda x: preprocess_batched(x, tokenizer, max_length),
                pad_idx,
                device,
                device_batch_size,
            )

            model.train()
            
            total_loss_value = total_loss.detach().cpu().item()
            perplexity = np.exp(total_loss_value)
            
            eval_losses.append((total_loss_value, update_step))
            eval_perplexities.append((perplexity, update_step))
            
            logger.info(
                f"[Eval Step {update_step}] Loss: {total_loss_value:.4f}, "
                f"PPL: {perplexity:.2f}, Eval tokens {evaluated_on_tokens}"
            )
    
    logger.info(f"Training finished for {optimizer_name}")
    
    return {
        'optimizer': optimizer_name,
        'train_losses': train_losses,
        'eval_losses': eval_losses,
        'eval_perplexities': eval_perplexities,
    }


# Main Experiment Runner
def run_experiment(optimizer_name, lr, device, output_dir):
    """
    Run a single training experiment with specified optimizer and learning rate.
    
    Args:
        optimizer_name: One of 'adam', 'rmsprop', 'sgd_momentum'
        lr: Learning rate
        device: Device to train on
        output_dir: Directory to save results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running experiment: {optimizer_name} with lr={lr}")
    logger.info(f"{'='*80}\n")
    
    # Configuration
    device_batch_size = 64
    max_length = 256
    num_training_steps = 5000 # so that training budget is 330M tokens
    total_batch_size = 256
    grad_clipping = 0.0
    print_freq = 100
    eval_every = 1000
    warmup_steps = 1000
    weight_decay = 0.0
    workers = 0
    
    gradient_accumulation = total_batch_size // device_batch_size
    
    # Load data via streaming
    logger.info("Loading C4 with streaming=True ...")

    data = load_dataset(
        "allenai/c4", "en", split="train", streaming=True
    )
    # val_data = load_dataset(
    #     "allenai/c4", "en", split="validation", streaming=True
    # )
    from datasets import load_from_disk
    val_data = load_from_disk("./data/c4/validation")

    logger.info("C4 streaming dataset loaded successfully.")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    pad_idx = tokenizer.pad_token_id
    
    # Preprocessed iterable dataset
    class PreprocessedIterableDataset(IterableDataset):
        def __init__(self, data, tokenizer, device_batch_size, max_length):
            super().__init__()
            self.data = data
            self.tokenizer = tokenizer
            self.device_batch_size = device_batch_size
            self.max_length = max_length

        def __iter__(self):
            iter_data = iter(self.data)
            batch = []

            for example in iter_data:
                tokenized = self.tokenizer(
                    example["text"],
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                batch.append(tokenized)
                if len(batch) == self.device_batch_size:
                    yield self._format_batch(batch)
                    batch = []

            if batch:
                yield self._format_batch(batch)

        def _format_batch(self, batch):
            input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
            return input_ids

    # create dataset and dataloader        
    dataset = PreprocessedIterableDataset(
        data,
        tokenizer,
        device_batch_size=device_batch_size,
        max_length=max_length,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=workers,
    )
    
    # model configuration
    logger.info("Initializing model...")
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0
    bias = False
    block_size = 1024
    vocab_size = tokenizer.vocab_size
    
    model_args = dict(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
        bias=bias, vocab_size=vocab_size, dropout=dropout
    )
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(device)
    model = torch.compile(model) # for faster training
    # model = model
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # optimizer
    logger.info(f"Initializing {optimizer_name} optimizer with lr={lr}...")
    
    if optimizer_name == 'adam':
        optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSProp(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd_momentum':
        optimizer = SGDMomentum(trainable_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=-1,
    )
    
    training_config = {
        'num_training_steps': num_training_steps,
        'gradient_accumulation': gradient_accumulation,
        'grad_clipping': grad_clipping,
        'print_freq': print_freq,
        'eval_every': eval_every,
        'pad_idx': pad_idx,
        'device_batch_size': device_batch_size,
        'max_length': max_length,
    }
    
    # Train
    model.train()
    history = train_model(
        optimizer_name=f"{optimizer_name}_lr{lr}",
        optimizer=optimizer,
        scheduler=scheduler,
        model=model,
        dataloader=dataloader,
        val_data=val_data,
        tokenizer=tokenizer,
        device=device,
        config=training_config
    )
    
    # save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / f"{optimizer_name}_lr{lr}_results.json"
    with open(result_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Results saved to {result_file}")
    
    # clean up memory
    # delete model and optimizer to free GPU memory
    del model
    del optimizer
    del scheduler
    del dataloader
    del dataset
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    logger.info("Cleaned up GPU memory")
    
    return history


# Main: run all experiments
def main():
    """
    Main function to run all experiments.
    
    This will run 9 experiments total:
    - 3 optimizers (Adam, RMSProp, SGD with Momentum)
    - 3 learning rates for each optimizer
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    output_dir = "./experiments"
    
    logger.info(f"Using device: {device}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Output directory (absolute): {os.path.abspath(output_dir)}")
    logger.info(f"Output directory (relative): {output_dir}")
    
    # Define learning rates to test for each optimizer
    # These are chosen based on typical ranges for each optimizer
    learning_rates = {
        'adam': [1e-4, 3e-4, 1e-3],           # Adam typically uses 1e-4 to 1e-3
        'rmsprop': [1e-4, 5e-4, 1e-3],        # RMSprop similar to Adam
        'sgd_momentum': [1e-3, 5e-3, 1e-2],   # SGD often needs higher learning rates
    }
    
    all_results = []
    
    # Run all experiments
    for optimizer_name in ['adam', 'rmsprop', 'sgd_momentum']:
        for lr in learning_rates[optimizer_name]:
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"Starting experiment {len(all_results)+1}/9: {optimizer_name} lr={lr}")
                logger.info(f"{'='*80}\n")
                
                history = run_experiment(optimizer_name, lr, device, output_dir)
                all_results.append(history)
                
                # Extra memory cleanup between experiments
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                logger.info(f"Completed experiment {len(all_results)}/9")
                logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
                
            except Exception as e:
                logger.error(f"Error in experiment {optimizer_name} lr={lr}: {e}")
                logger.error(f"Continuing to next experiment...")
                
                # Clean up even on error
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                continue
    
    logger.info("\n" + "="*80)
    logger.info("ALL EXPERIMENTS COMPLETED!")
    logger.info("="*80)


if __name__ == "__main__":
    main()