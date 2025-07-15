import torch
import glob
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm.auto import tqdm 
from moe_nano_gpt_model import NanoGPTMoE

# --- CUDA setup ---
torch.cuda.memory._record_memory_history()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

# --- Load Dataset and Tokenizer ---
dataset = load_dataset("roneneldan/TinyStories")
tokenizer = Tokenizer.from_file("data/TinyStories-tokenizer.json")

# --- Hyperparameters ---
hyperparameters = {
    "n_epochs": 5,
    "vocab_size": tokenizer.get_vocab_size(),
    "batch_size": 32,
    "block_size": 512,
    "learning_rate": 1e-5,
    "n_embed": 256,
    "n_heads": 6,
    "n_layers": 6,
    "dropout": 0.2,
    "n_experts": 8,
    "top_k": 2,
}
n_epochs = hyperparameters["n_epochs"]
batch_size = hyperparameters["batch_size"]
block_size = hyperparameters["block_size"]
learning_rate = hyperparameters["learning_rate"]

# --- Tokenize ---
tokenizer.enable_padding(pad_id=2, pad_token="<|im_end|>", length=block_size)
tokenizer.enable_truncation(max_length=block_size)
tokenized_data = dataset.map(
    lambda x: {"input_ids": [elem.ids for elem in tokenizer.encode_batch(x["text"])]},
    batched=True,
)
tokenized_data = tokenized_data.with_format("torch")
train_ids = tokenized_data["train"].remove_columns(["text"]).shuffle().select(range(30000))
val_ids = tokenized_data["validation"].remove_columns(["text"]).shuffle().select(range(3000))

# --- Model, Dataloader, Optimizer ---
model = NanoGPTMoE(hyperparameters, device).to(device)
train_dataloader = DataLoader(
    train_ids,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True
)

val_dataloader = DataLoader(
    val_ids,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True
)


optimizer = AdamW(model.parameters(), lr=learning_rate)
num_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"{num_params:.3f}M parameters")

# --- Load Checkpoint if Available ---
def get_latest_checkpoint(folder="checkpoints2"):
    checkpoints = glob.glob(f"{folder}/moe-*.pt")
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

checkpoint_file = get_latest_checkpoint()
saved_epoch = None

if checkpoint_file is not None and os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    saved_epoch = checkpoint["epoch"]
    print(f"Loaded checkpoint from epoch {saved_epoch+1}")
else:
    print("No checkpoint found. Starting from scratch.")

# --- Scheduler (don't reuse state) ---
num_training_steps = n_epochs * len(train_dataloader)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=learning_rate, T_max=len(train_dataloader) * n_epochs)

# --- Training Loop ---
lossi, lri = [], []
global_step = 0

for epoch in range(n_epochs):
    if saved_epoch is not None and epoch <= saved_epoch:
        continue

    model.train()
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{n_epochs}")
    for step, batch in progress_bar:
        batch = batch["input_ids"].to(device)
        targets = torch.concat((batch[:, 1:], 2 * torch.ones([batch.shape[0], 1]).to(device)), dim=1).long()

        logits, loss = model(batch, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        lossi.append(loss.log10().item())
        lri.append(optimizer.param_groups[0]["lr"])
        global_step += 1

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        val_losses = torch.zeros(len(val_dataloader), device=device)
        for k, batch in enumerate(val_dataloader):
            batch = batch["input_ids"].to(device)
            targets = torch.concat((batch[:, 1:], 2 * torch.ones([batch.shape[0], 1]).to(device)), dim=1).long()
            logits, loss = model(batch, targets)
            val_losses[k] = loss.item()
        avg_val_loss = val_losses.mean()
        print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")

        train_losses = torch.zeros(len(val_dataloader), device=device)
        for k, batch in enumerate(train_dataloader):
            if k == len(val_dataloader):
                break
            batch = batch["input_ids"].to(device)
            targets = torch.concat((batch[:, 1:], 2 * torch.ones([batch.shape[0], 1]).to(device)), dim=1).long()
            logits, loss = model(batch, targets)
            train_losses[k] = loss.item()
        avg_train_loss = train_losses.mean()
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")

        # --- Save Checkpoint ---
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "hyperparameters": hyperparameters,
            "val_loss": avg_val_loss.item(),
            "train_loss": avg_train_loss.item(),
            "lossi": lossi,
            "lri": lri,
        }
        torch.save(checkpoint, f"checkpoints2/moe-{num_params:.3f}M-checkpoint-{epoch}.pt")
