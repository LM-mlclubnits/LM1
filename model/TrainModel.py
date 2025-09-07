import torch
import torch.nn as nn
import streamlit as st


def train_model(model, data, tokenizer, epochs=200, lr=1e-3):
    """
    model: a nn.Module (already moved to desired device)
    data: list of strings
    tokenizer: our BPE Tokenizer wrapper instance
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Prepare inputs/targets as token id lists
    inputs = [torch.tensor(tokenizer.encode(t)[:-1], dtype=torch.long) for t in data]
    targets = [torch.tensor(tokenizer.encode(t)[1:], dtype=torch.long) for t in data]

    # Determine device from model parameters
    device = next(model.parameters()).device

    loss_p = st.empty()
    chart_p = st.empty()
    loss_vals = []

    for epoch in range(epochs):
        tot_loss = 0
        # shuffle per epoch for better generalization
        perm = torch.randperm(len(inputs))
        for i in perm:
            inp = inputs[i]
            tgt = targets[i]

            # move to device, add batch dim
            inp, tgt = inp.unsqueeze(0).to(device), tgt.unsqueeze(0).to(device)

            optimizer.zero_grad()
            logits = model(inp)  # returns (B, T, vocab_size)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

        avg_loss = tot_loss / len(inputs)
        loss_vals.append(avg_loss)

        loss_p.text(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        chart_p.line_chart(loss_vals)
