import torch
import torch.nn as nn
from .transformer import TransformerBlock


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, d_ff=2048, n_layers=6, max_len=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.embed(x) + self.pos_embed(pos)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)

        for block in self.blocks:
            x = block(x, mask=mask)
            x = self.ln(x)
        return self.fc(x)

    def generate(self, idx, max_new_tokens) -> int:
        for _ in range(max_new_tokens):
            logits = self.forward(idx)
            next_token = torch.argmax(logits[:,-1,:], dim=-1).unsqueeze(1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
