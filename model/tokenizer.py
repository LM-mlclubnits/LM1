"""
BPE Tokenizer wrapper using HuggingFace tokenizers (ByteLevelBPETokenizer).

Install dependency:
    pip install tokenizers

This Tokenizer class exposes:
- Tokenizer(texts=None, vocab_size=5000, save_dir="./tokenizer")  # if texts passed and tokenizer files missing, it will train and save
- encode(text) -> list[int]
- decode(list_of_ids) -> str
- get_vocab_size() -> int
"""

import os
import json
from tokenizers import ByteLevelBPETokenizer


class Tokenizer:
    def __init__(self, texts=None, vocab_size=5000, save_dir="./tokenizer"):
        """
        If tokenizer files exist in save_dir (vocab.json & merges.txt), they are loaded.
        Otherwise, if `texts` is provided (list of strings), the tokenizer will be trained and saved.
        """
        self.save_dir = save_dir
        self.vocab_file = os.path.join(save_dir, "vocab.json")
        self.merges_file = os.path.join(save_dir, "merges.txt")

        if os.path.exists(self.vocab_file) and os.path.exists(self.merges_file):
            # load existing
            self._tokenizer = ByteLevelBPETokenizer(self.vocab_file, self.merges_file)
        else:
            if texts is None:
                raise ValueError("No tokenizer files found and no `texts` supplied to train a new tokenizer.")
            # write texts to a temporary training file
            train_path = "bpe_train.txt"
            with open(train_path, "w", encoding="utf-8") as f:
                for line in texts:
                    f.write(line.replace("\n", " ") + "\n")
            self._tokenizer = ByteLevelBPETokenizer()
            special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
            self._tokenizer.train(files=train_path, vocab_size=vocab_size, min_frequency=1, special_tokens=special_tokens)
            os.makedirs(save_dir, exist_ok=True)
            self._tokenizer.save_model(save_dir)
            try:
                os.remove(train_path)
            except Exception:
                pass

        # basic tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

    def encode(self, text):
        """
        Returns a list of token ids (ints).
        """
        enc = self._tokenizer.encode(text)
        return enc.ids

    def decode(self, ids):
        """
        ids: list[int]
        returns: decoded string
        """
        # byte-level tokenizer's decode accepts list of ids
        return self._tokenizer.decode(ids)

    def get_vocab_size(self):
        """
        Returns vocabulary size (int).
        """
        # Read vocab.json and count entries (stable across tokenizers)
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            return len(vocab)
        # fallback to tokenizer API (if present)
        try:
            return self._tokenizer.get_vocab_size()
        except Exception:
            # as a final fallback, attempt to extract vocab dict
            try:
                return len(self._tokenizer.get_vocab())
            except Exception:
                raise RuntimeError("Unable to determine tokenizer vocabulary size.")
