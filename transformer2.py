import argparse
import base64
import json
import random
import struct
import torch
from typing import Literal

class DataSlurper:
    def __init__(self, filename_base: str, device: str):
        self.files = {
            'train': open(f'{filename_base}.train','rb'),
            'validation': open(f'{filename_base}.validation','rb'),
        }
        with open(f'{filename_base}.metadata') as f:
            self.metadata = json.load(f)
        self.device = device

    def _pick_articles(self, split: Literal['train','validation'], n: int) -> list:
        return random.sample(self.metadata[split], n)

    def batch(self, split: Literal['train','validation'], n: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a uint16 tensor of shape (n, length) containing the first `length` tokens of `n` random articles from the specified split.
        Each row will start with a zero byte which is the beginning-of-sequence token.

        Also returns a bool tensor of shape (n, length) containing the part of the tensor that is actually filled with data.
        (It may be padded with zeros in the case that the end of the article is reached).
        """
        byte_length = 2 * (length - 1)
        articles = self._pick_articles(split, n)
        result = torch.zeros((n, length), dtype=torch.uint16)
        result_mask = torch.zeros((n, length), dtype=bool)
        for i,article in enumerate(articles):
            f = self.files[split]
            start = article['token_start']
            end = min(article['token_end'], start + byte_length)
            f.seek(start)
            read_bytes = f.read(end-start)
            read_tensor = torch.frombuffer(read_bytes, dtype=torch.int16)
            result[i,1:1+len(read_tensor)] = read_tensor
            result_mask[i,:1+len(read_tensor)] = True
        return result.to(self.device), result_mask.to(self.device)

class Tokenizer:
    def __init__(self, dict_filename: str):
        self.tokens = [b'[--SEP--]']
        with open(dict_filename) as f:
            for line in f:
                self.tokens.append(base64.b64decode(line.strip()))
    
    def decode(self, tensor: torch.Tensor) -> str:
        if len(tensor.shape) != 1:
            raise ValueError('Expected a 1D tensor')
        return b''.join(self.tokens[t] for t in tensor).decode('utf-8', errors='replace')

class TransformerModel(torch.nn.Module):
    def __init__(self, n_layers: int, n_heads: int, n_dict: int, d_model: int, d_k: int, d_hidden: int):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.randn(n_dict, d_model))
        self.wk = torch.nn.Parameter(torch.randn(n_layers, n_heads, d_model, d_k))
        self.wq = torch.nn.Parameter(torch.randn(n_layers, n_heads, d_model, d_k))
        self.wv = torch.nn.Parameter(torch.randn(n_layers, n_heads, d_model, d_k))
        self.mlp0 = torch.nn.Parameter(torch.randn(n_layers, d_model, d_hidden))
        self.mlp1 = torch.nn.Parameter(torch.randn(n_layers, d_hidden, d_model))
        self.unembedding = torch.nn.Parameter(torch.randn(d_model, n_dict))
        self.n_layers = n_layers
        self.n_heads = n_heads
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding[x]
        print(x.shape)
        for layer in range(self.n_layers):
            k = torch.einsum('btm,hmk->bhtk', x, self.wk[layer])
            q = torch.einsum('btm,hmk->bhtk', x, self.wq[layer])
            v = torch.einsum('btm,hmk->bhtk', x, self.wv[layer])
            attn = torch.einsum('bhtk,bhTk->bhtT', q, k)
            print(attn.shape)
            print(attn)
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('-d', type=str, required=True)
    args = parser.parse_args()
    input_filename = args.input_file
    dict_filename = args.d
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    slurper = DataSlurper(input_filename, device)
    tokenizer = Tokenizer(dict_filename)
    
    n_batch = 64
    n_context = 128
    d_model = 512
    d_k = 64
    d_hidden = 1024

    model = TransformerModel(2, 8, len(tokenizer.tokens), d_model, d_k, d_hidden).to(device)

    batch, mask = slurper.batch('train', n_batch, n_context)
    for row in batch:
        print(tokenizer.decode(row))

    model(batch)

if __name__ == '__main__':
    main()