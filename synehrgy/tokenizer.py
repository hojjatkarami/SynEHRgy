import json
from pathlib import Path

# from typing import Dict, List
from transformers import PreTrainedTokenizer
import os

class EHRTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_list):
        # Load vocabulary from file
        self.vocab = {
            # "[PAD]": 0,
            "[UNK]": 0,
            # "[SEP]": 2,  # Do not declare self.sep_token as it increases training loss
        }
        self.vocab |= {
            token: i + (1 + max(self.vocab.values()))
            for i, token in enumerate(vocab_list)
            if token
        }

        self.id_to_token = {v: k for k, v in self.vocab.items()}  # Reverse mapping

        # Special tokens
        self.unk_token = "[UNK]"
        self.pad_token = "<pad>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        # self.sep_token = "[SEP]"

        super().__init__(
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            # sep_token=self.sep_token,
            # vocab_list=vocab_list,
        )

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self.id_to_token.get(index, self.unk_token)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    # def save_vocabulary(
    #     self, save_directory: str, filename="vocab.json", filename_prefix=None
    # ):
    #     if filename_prefix:
    #         filename = f"{filename_prefix}_{filename}"
    #     path = f"{save_directory}/{filename}"
    #     with open(path, "w", encoding="utf-8") as f:
    #         json.dump(self.vocab, f, ensure_ascii=False, indent=2)
    #     return (path,)

    def save_vocabulary(
        self, save_directory: str, filename="vocab.json", filename_prefix=None
    ):
        if filename_prefix:
            filename = f"{filename_prefix}_{filename}"
        path = os.path.join(save_directory, filename)

        # Convert tuple keys to JSON strings (reversible)
        vocab_to_save = {
            json.dumps(k) if isinstance(k, tuple) else k: v
            for k, v in self.vocab.items()
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(vocab_to_save, f, ensure_ascii=False, indent=2)

        return (path,)

    def _tokenize(self, text: str) -> list[str]:
        # Input is a single string (not a list), so wrap it in a list
        return [text]

    def add_tokens(self, new_tokens: list[str]) -> int:
        """
        Add new tokens to the vocabulary.

        Returns the number of tokens actually added (i.e., not already
        in vocab).
        """
        added = 0
        start_idx = max(self.vocab.values()) + 1 if self.vocab else 0

        for token in new_tokens:
            if token not in self.vocab:
                self.vocab[token] = start_idx
                self.id_to_token[start_idx] = token
                start_idx += 1
                added += 1

        self.total_vocab_size = len(self.get_vocab())

        return added

    @classmethod
    def from_pretrained(cls, save_directory, *args, **kwargs):
        vocab_path = Path(save_directory) / "vocab.json"
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)
        vocab_list = [
            tok for tok in vocab.keys() if tok not in ["[PAD]", "[UNK]", "[SEP]"]
        ]
        return cls(vocab_list=vocab_list, **kwargs)