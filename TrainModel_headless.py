import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
import time

def create_dataloader(data, tokenizer, batch_size=16):
    inputs = [torch.tensor(tokenizer.encode(t)[:-1], dtype=torch.long) for t in data]
    targets = [torch.tensor(tokenizer.encode(t)[1:], dtype=torch.long) for t in data]
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=tokenizer.stoi["<pad>"])
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=tokenizer.stoi["<pad>"])
    dataset = TensorDataset(padded_inputs, padded_targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

@torch.no_grad()
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    for inp, tgt in val_loader:
        inp, tgt = inp.to(device), tgt.to(device)
        logits = model(inp)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
        total_loss += loss.item()
    model.train()
    return total_loss / len(val_loader)

def train_model_headless(model, data, tokenizer, epochs=50, lr=3e-4, batch_size=16, device="cpu"):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi["<pad>"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_size = int(0.9 * len(data))
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size])
    train_loader = create_dataloader(train_data, tokenizer, batch_size=batch_size)
    val_loader = create_dataloader(val_data, tokenizer, batch_size=batch_size)
    print("--- Starting Training ---")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_train_loss = 0
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()
            logits = model(inp)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Duration: {epoch_duration:.2f}s")
    print("--- Training Complete ---")
    return model