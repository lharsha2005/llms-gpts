import torch
from gptV2 import NanoGPT
from tokenizers import Tokenizer

# -------- Setup --------
torch.cuda.memory._record_memory_history()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

# -------- Load Tokenizer --------
tokenizer = Tokenizer.from_file("tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

# -------- Load Checkpoint --------
checkpoint = torch.load("checkpoints/checkpoint-5.pt")
hyperparameters = checkpoint['hyperparameters']
model = NanoGPT(hyperparameters, device).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

# -------- Take User Input --------
user_text = input("Enter your prompt: ")
encoded = tokenizer.encode(user_text)
context = torch.tensor([encoded.ids], dtype=torch.long, device=device)

# -------- Generate Text --------
with torch.no_grad():
    generated = model.generate(context, max_new_tokens=256)[0]

# -------- Decode and Print Output --------
output_text = tokenizer.decode(generated.tolist())
print("\nGenerated Text:\n" + output_text)
