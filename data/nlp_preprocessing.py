import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List


class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        words = set()
        for pos, neg in zip(positive, negative):
            words.update(set(pos.split()))
            words.update(set(neg.split()))
            
        words = sorted(list(words))

        w_map = {}
        for i, w in enumerate(words):
            w_map[w] = i + 1

        all_sentences = positive + negative
        # Now iterate through all_sentences to convert each to a tensor
        tensors = [torch.tensor([w_map[w] for w in s.split()]) for s in all_sentences]
        return nn.utils.rnn.pad_sequence(tensors, batch_first=True)
