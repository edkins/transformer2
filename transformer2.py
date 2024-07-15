import argparse
import base64
import torch
from typing import Literal

class DataSlurper:
    def __init__(self, filename_base: str, split: Literal['train','validation','test'], device: str, n_batch: int, n_context: int):
        self.split = split
        self.file = open(f'{filename_base}.{split}','rb')
        with open(f'{filename_base}.metadata.{split}','rb') as f:
            buffer = f.read()
            self.metadata = torch.frombuffer(buffer, dtype=torch.int64).reshape(-1,2)
        self.device = device
        self.n_batch = n_batch
        self.n_context = n_context

    def _pick_articles(self, n: int) -> torch.Tensor:
        indices = torch.randint(0, len(self.metadata), (n,))
        return self.metadata[indices]

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a uint16 tensor of shape (n, length) containing the first `length` tokens of `n` random articles from the specified split.
        Each row will start with a zero byte which is the beginning-of-sequence token.

        Also returns a bool tensor of shape (n, length) containing the part of the tensor that is actually filled with data.
        (The data may be padded with zeros in the case that the end of the article is reached).
        """
        with torch.no_grad():
            byte_length = 2 * (self.n_context - 1)
            articles = self._pick_articles(self.n_batch)
            result = torch.zeros((self.n_batch, self.n_context), dtype=torch.uint16)
            result_mask = torch.zeros((self.n_batch, self.n_context), dtype=bool)
            for i,[start,token_end] in enumerate(articles):
                end = min(token_end, start + byte_length)
                self.file.seek(start)
                read_bytes = self.file.read(end-start)
                read_tensor = torch.frombuffer(read_bytes, dtype=torch.int16)
                result[i,1:1+len(read_tensor)] = read_tensor
                result_mask[i,:len(read_tensor)] = True
            return result.to(self.device), result_mask.to(self.device)

class Tokenizer:
    def __init__(self, dict_filename: str):
        self.tokens = [b'$']    # start/end of document marker. Not an actual dollar sign.
        with open(dict_filename) as f:
            for line in f:
                self.tokens.append(base64.b64decode(line.strip()))
    
    def decode(self, tensor: torch.Tensor) -> str:
        if len(tensor.shape) != 1:
            raise ValueError('Expected a 1D tensor')
        return b''.join(self.tokens[t] for t in tensor).decode('utf-8', errors='replace')

def param(*size):
    return torch.nn.Parameter(torch.randn(size) / 20)

class TransformerModel(torch.nn.Module):
    def __init__(self, n_layer: int, n_head: int, n_dict: int, d_model: int, d_k: int, d_hidden: int):
        super().__init__()
        self.embedding = param(n_dict, d_model)
        self.wq = param(n_layer, n_head, d_model, d_k)
        self.wk = param(n_layer, n_head, d_model, d_k)
        self.wv = param(n_layer, n_head, d_model, d_model)
        self.mlp0 = param(n_layer, d_model, d_hidden)
        self.mlp1 = param(n_layer, d_hidden, d_model)
        self.unembedding = param(d_model, n_dict)
        self.bias = param(n_dict)
        self.n_layer = n_layer
        self.n_head = n_head
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.embedding(x.long(), self.embedding)
        for layer in range(self.n_layer):
            # attention
            q = torch.einsum('btm,hmq->bhtq', x, self.wq[layer])
            k = torch.einsum('bom,hmk->bhok', x, self.wk[layer])
            v = torch.einsum('bom,hmv->bhov', x, self.wv[layer])
            attn = torch.einsum('bhtq,bhoq->bhto', q, k)
            attn = torch.exp(attn)
            attn = torch.tril(attn)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)
            x = x + torch.einsum('bhto,bhov->btv', attn, v)

            # mlp
            y = torch.einsum('btm,mh->bth', x, self.mlp0[layer])
            y = torch.relu(y)
            y = torch.einsum('bth,hm->btm', y, self.mlp1[layer])
            x = x + y

        return torch.einsum('btm,md->btd', x, self.unembedding) + self.bias

def validation(model, batch, mask, tokenizer):
    with torch.no_grad():
        y = model(batch)
        mr = mask[:,1:].reshape(-1)
        yr = y[:,:-1,:].reshape(-1, y.shape[-1])[mr]
        br = batch[:,1:].reshape(-1).long()[mr]
        loss = torch.nn.functional.cross_entropy(yr, br, reduction='mean')
        print(f'  Validation loss: {loss.item()}')
        #print(tokenizer.decode(batch[0]))

def train(model, slurper, n_batches, vbatch, vmask, tokenizer):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(n_batches):
        batch, mask = slurper.batch()
        y = model(batch)
        mr = mask[:,1:].reshape(-1)
        yr = y[:,:-1,:].reshape(-1, y.shape[-1])[mr]
        br = batch[:,1:].reshape(-1).long()[mr]
        loss = torch.nn.functional.cross_entropy(yr, br, reduction='mean')
        loss.backward()
        opt.step()
        opt.zero_grad()
        if (i + 1) % 1000 == 0:
            print(f'Batch {i+1}, loss: {loss.item()}')
            #print(tokenizer.decode(batch[0]))
            validation(model, vbatch, vmask, tokenizer)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    args = parser.parse_args()
    input_filename = args.input_file
    dict_filename = f'{input_filename}.dictionary'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = Tokenizer(dict_filename)
    
    n_layer = 1
    n_head = 2
    n_dict = len(tokenizer.tokens)
    n_batch = 1
    n_context = 128
    d_model = 32
    d_k = 8
    d_hidden = 1024

    torch.manual_seed(12345)
    vslurper = DataSlurper(input_filename, 'validation', device, 64, n_context)
    slurper = DataSlurper(input_filename, 'train', device, n_batch, n_context)
    vbatch, vmask = vslurper.batch()
    model = TransformerModel(n_layer, n_head, n_dict, d_model, d_k, d_hidden).to(device)
    train(model, slurper, 20000, vbatch, vmask, tokenizer)

if __name__ == '__main__':
    main()