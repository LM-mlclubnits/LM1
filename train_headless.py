import torch
from stories_list import stories
from tokenizer import Tokenizer
from decoder import Decoder
from TrainModel_headless import train_model_headless

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data and initializing tokenizer...")
    tokenizer = Tokenizer(stories)
    vocab_size = len(tokenizer.stoi)
    print(f"Data loaded: {len(stories)} stories.")
    print(f"Vocabulary size: {vocab_size}")

    print("Initializing model...")
    D_MODEL, N_HEADS, D_FF, N_LAYERS, DROPOUT = 256, 4, 1024, 4, 0.1
    EPOCHS, BATCH_SIZE, LEARNING_RATE = 50, 16, 3e-4
    
    model = Decoder(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        max_len=4096
    )

    model.to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    trained_model = train_model_headless(
        model=model,
        data=stories,
        tokenizer=tokenizer,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        device=device
    )

    model_save_path = "story_model.pth"
    print(f"Saving trained model to {model_save_path}...")
    torch.save(trained_model.state_dict(), model_save_path)
    print("Model saved.")

    print("\n--- Generating a sample story ---")
    prompt = "once upon a time"
    encoded_prompt = torch.tensor(tokenizer.encode(prompt.lower()), dtype=torch.long).unsqueeze(0).to(device)
    
    generated_ids = trained_model.generate(encoded_prompt, max_new_tokens=100)
    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
    print(f"Prompt: '{prompt}'")
    print(f"Generated Text: \n{generated_text}")

if __name__ == "__main__":
    run()