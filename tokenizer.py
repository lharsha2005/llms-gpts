from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC

tokenizer = Tokenizer(BPE())
tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = Whitespace()

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = BpeTrainer(vocab_size=30000, special_tokens=special_tokens)

tokenizer.train(["bio_data.txt"], trainer)

tokenizer.save("tokenizer.json")