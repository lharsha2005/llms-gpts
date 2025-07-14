import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from tokenizers import Tokenizer
from gptV2 import NanoGPT
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import os
import random

torch.cuda.memory._record_memory_history()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

# -------- Dataset --------
class GPTDataset(torch.utils.data.Dataset):
    def __init__(self, chunks):
        self.samples = chunks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long)

# -------- Load Data --------
with open("bio_data.txt", "r", encoding="utf-8") as file:
    data = file.read()
data = data.replace("\n", " ").strip()

tokenizer = Tokenizer.from_file("tokenizer.json")
eos_token_id = tokenizer.token_to_id("[SEP]")
enc = tokenizer.encode(data)
ids = enc.ids

# -------- Hyperparameters --------
hyperparameters = {
    "n_epochs": 20,
    "vocab_size": tokenizer.get_vocab_size(),
    "batch_size": 8,
    "block_size": 1080,
    "learning_rate": 5e-4,
    "n_embed": 516,
    "n_heads": 8,
    "n_layers": 6,
    "dropout": 0.2,
}

stride = hyperparameters["block_size"] // 2
chunks = [ids[i:i + hyperparameters["block_size"]] for i in range(0, len(ids) - hyperparameters["block_size"], stride)]
random.shuffle(chunks)
train_split = int(0.9 * len(chunks))
train_dataset = GPTDataset(chunks[:train_split])
val_dataset = GPTDataset(chunks[train_split:])

train_dataloader = DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=hyperparameters["batch_size"], shuffle=False)

# -------- Model & Optimizer --------
model = NanoGPT(hyperparameters, device).to(device)
optimizer = AdamW(model.parameters(), lr=hyperparameters["learning_rate"])

num_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"{num_params:.3f}M parameters")

num_training_steps = hyperparameters["n_epochs"] * len(train_dataloader)
scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=hyperparameters["learning_rate"], total_steps=num_training_steps)

# -------- Training Loop --------
saved_epoch = None
lossi = []
lri = []

global_step = 0
progress_bar = tqdm(total=num_training_steps)

for epoch in range(hyperparameters["n_epochs"]):
    model.train()
    if saved_epoch is not None and epoch <= saved_epoch:
        continue

    for batch in train_dataloader:
        batch = batch.to(device)
        eos_column = eos_token_id * torch.ones((batch.size(0), 1), device=device, dtype=torch.long)
        targets = torch.cat((batch[:, 1:], eos_column), dim=1)
        logits, loss = model(batch, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        lossi.append(loss.log10().item())
        lri.append(optimizer.param_groups[0]['lr'])

        global_step += 1
        progress_bar.update(1)

        progress_bar.set_postfix({
            "train_loss": f"{loss.item():.4f}",
        })

    # -------- Validation --------
    with torch.no_grad():
        model.eval()

        val_losses = torch.zeros(len(val_dataloader), device=device)
        for k, batch in enumerate(val_dataloader):
            batch = batch.to(device)
            targets = torch.cat((batch[:, 1:], 2 * torch.ones([batch.shape[0], 1], device=device)), dim=1).long()
            _, loss = model(batch, targets)
            val_losses[k] = loss.item()

        avg_val_loss = val_losses.mean().item()

        train_losses = torch.zeros(len(val_dataloader), device=device)
        for k, batch in enumerate(train_dataloader):
            if k == len(val_dataloader):  # match number of steps
                break
            batch = batch.to(device)
            targets = torch.cat((batch[:, 1:], 2 * torch.ones([batch.shape[0], 1], device=device)), dim=1).long()
            _, loss = model(batch, targets)
            train_losses[k] = loss.item()

        avg_train_loss = train_losses.mean().item()

        # Update progress bar with val/train loss
        progress_bar.set_postfix({
            "train_loss": f"{avg_train_loss:.4f}",
            "val_loss": f"{avg_val_loss:.4f}",
        })

        # -------- Save Checkpoint --------
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "hyperparameters": hyperparameters,
            "val_loss": avg_val_loss,
            "train_loss": avg_train_loss,
        }
        torch.save(checkpoint, f"checkpoints/checkpoint-{epoch}.pt")

progress_bar.close()
