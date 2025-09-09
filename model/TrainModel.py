import torch
import torch.nn as nn
import streamlit as st


def train_model(model, data, tokenizer, epochs=200, lr=1e-3):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0) #to ignore padding

    inputs = [torch.tensor(tokenizer.encode(t)[:-1], dtype=torch.long) for t in data]
    targets = [torch.tensor(tokenizer.encode(t)[1:], dtype=torch.long) for t in data]

    loss_p = st.empty()
    chart_p = st.empty()
    loss_vals = []



    for epoch in range(epochs):
        tot_loss = 0
        for inp, tgt in zip(inputs, targets):
            inp, tgt = inp.unsqueeze(0), tgt.unsqueeze(0)
            optimizer.zero_grad()
            logits = model(inp)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

        avg_loss = tot_loss / len(inputs)
        loss_vals.append(avg_loss)

        loss_p.text(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        chart_p.line_chart(loss_vals)
