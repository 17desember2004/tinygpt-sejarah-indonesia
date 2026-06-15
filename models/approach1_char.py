"""
APPROACH 1: CHARACTER-LEVEL TOKENIZATION
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_blocks import Block

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Read corpus
with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("="*70)
print("APPROACH 1: CHARACTER-LEVEL TOKENIZATION")
print("="*70)

# Build character vocabulary
chars = sorted(set(text))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}
vocab_size_char = len(chars)

print(f"\n✓ Vocab size: {vocab_size_char}")
print(f"✓ Corpus: {len(text)} characters, {len(text.split())} words\n")

# Encode text
char_tokens = [char2idx[c] for c in text]
data_char = torch.tensor(char_tokens, dtype=torch.long)

# Model definition
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size=6, embedding_dim=32, n_heads=2, n_layers=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)
        self.block_size = block_size
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

block_size = 6
embedding_dim = 32
n_heads = 2
n_layers = 2
lr = 1e-3
epochs = 1500
batch_size = 16

def get_batch(data, block_size, batch_size=16):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Initialize model
model_char = TinyGPT(vocab_size_char, block_size, embedding_dim, n_heads, n_layers).to(device)
optimizer = torch.optim.AdamW(model_char.parameters(), lr=lr)

# Training loop
print(f" Training Character-Level Model ({epochs} epochs)...\n")
losses_char = []

for step in range(epochs):
    xb, yb = get_batch(data_char, block_size, batch_size)
    logits, loss = model_char(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses_char.append(loss.item())
    
    if step % 300 == 0:
        print(f"Step {step:4d}, loss={loss.item():.4f}")

print(f"\n Training selesai!")
print(f"Initial Loss: {losses_char[0]:.4f}")
print(f"Final Loss:   {losses_char[-1]:.4f}")
print(f"Loss Reduction: {(1 - losses_char[-1]/losses_char[0])*100:.1f}%")

# Generate text
print(f"\n Generated Text (Prompt 'Sejarah'):")
context = torch.tensor([[char2idx.get(c, 0) for c in "Sejarah"]], dtype=torch.long).to(device)
output = model_char.generate(context, max_new_tokens=100)
generated_text = ''.join([idx2char.get(idx, '?') for idx in output[0].tolist()])
print(f"Output: {generated_text[:100]}...")

# Save for comparison
import pickle
with open('approach1_char.pkl', 'wb') as f:
    pickle.dump({
        'name': 'Character-Level',
        'vocab_size': vocab_size_char,
        'losses': losses_char,
        'generated_text': generated_text
    }, f)

print("\n APPROACH 1 COMPLETE!\n")

