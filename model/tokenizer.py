
#hcbkasjcvbaksjhvbashfvbkvbkjasv

class Tokenizer:
    def __init__(self, texts):
        vocab = set(" ".join(texts).split())
        self.stoi = {i: w for w, i in enumerate(sorted(vocab), start=2)}
        self.stoi["<pad>"] = 0
        self.stoi["<unk>"] = 1
        self.itos = {i: w for w, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi.get(w, 1) for w in text.split()]

    def decode(self, ids):
        return " ".join([self.itos.get(i, "<unk>") for i in ids])
