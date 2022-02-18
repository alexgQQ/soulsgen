import os
import torch
from tokenization import Tokenizer
from vocabulary import Vocab
from transformer import Transformer
from typing import List


class GPT2GenerationModel(Transformer):

    vocab_path = os.environ.get("GPT2_VOCAB_PATH")
    seq_len = 64
    layers = 12
    heads = 16
    dims = 1024
    rate = 4

    def __init__(self):
        self.initialize()
        super(GPT2GenerationModel, self).__init__(
            layers=self.layers,
            pad_idx=self.vocab.pad_idx,
            words=len(self.vocab),
            seq_len=self.seq_len,
            heads=self.heads,
            dims=self.dims,
            rate=self.rate,
            dropout=0,
            bidirectional=False,
        )

    def initialize(self):
        self.vocab = Vocab(vocab_path=self.vocab_path)
        self.tokenizer = Tokenizer(vocab=self.vocab)

    def encode_context(self, context: str) -> List[int]:
        tokens = [self.vocab[t] for t in self.tokenizer.encode(context)]
        tokens = [self.vocab.bos_idx] + tokens

        return tokens

    def decode_tokens(self, tokens: List[int]) -> str:
        if self.vocab.eos_idx in tokens:
            tokens = tokens[: tokens.index(self.vocab.eos_idx) + 1]
        return self.tokenizer.decode([self.vocab[t] for t in tokens])

    def decorate_sequence(self, sequence: torch.Tensor, offset: int) -> torch.Tensor:
        return sequence
