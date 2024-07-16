import argparse
import base64
import json
import time
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
        print(f"Number of {split} articles: {len(self.metadata)}")

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
                result_mask[i,:1+len(read_tensor)] = True
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

def param(*size, mag=0.05):
    return torch.nn.Parameter(torch.randn(size, dtype=torch.float32) * mag)

def constant(value: float):
    return torch.nn.Parameter(torch.tensor(value, dtype=torch.float32), requires_grad=False)

class TransformerModel(torch.nn.Module):
    def __init__(self, n_layer: int, n_head: int, n_dict: int, d_model: int, d_k: int, d_hidden: int, n_context: int, mag: float, adiv: float, pdiv: float, fixedpos: Literal['True','False','None','FromZero'], layernorm: Literal['True','False','Affine'], enorm: Literal['True','False','Affine'], ldiv: float):
        super().__init__()
        self.embedding = param(n_dict, d_model, mag=mag)
        if fixedpos == 'True':
            pos_embedding = torch.zeros(1, n_context, d_model)
            # d_model must be even for this
            for i2 in range(0,d_model,2):
                pos_embedding[0,:,i2] = torch.sin(torch.arange(n_context, dtype=torch.float32) / (10000**(i2/d_model)))
                pos_embedding[0,:,i2+1] = torch.cos(torch.arange(n_context, dtype=torch.float32) / (10000**(i2/d_model)))
            self.pos_embedding = torch.nn.Parameter(pos_embedding, requires_grad=False)
        elif fixedpos == 'False':
            self.pos_embedding = param(1, n_context, d_model, mag=mag)
        elif fixedpos == 'None':
            pos_embedding = torch.zeros(1, n_context, d_model)
            self.pos_embedding = torch.nn.Parameter(pos_embedding, requires_grad=False)
        elif fixedpos == 'FromZero':
            pos_embedding = torch.zeros(1, n_context, d_model)
            self.pos_embedding = torch.nn.Parameter(pos_embedding)
        if d_k > 0:
            self.wq = param(n_layer, n_head, d_model, d_k, mag=mag)
            self.wk = param(n_layer, n_head, d_model, d_k, mag=mag)
            self.wv = param(n_layer, n_head, d_model, d_model, mag=mag)
        self.mlp0 = param(n_layer, d_model, d_hidden, mag=mag)
        self.mlpb = param(n_layer, 1, d_hidden, mag=mag)
        self.mlp1 = param(n_layer, d_hidden, d_model, mag=mag)
        self.unembedding = param(d_model, n_dict, mag=mag)
        self.bias = param(1,1,n_dict, mag=mag)
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_k = d_k
        self.amul = constant(1/adiv)
        self.pmul = constant(1/pdiv)
        self.layernorm = layernorm
        if layernorm == 'True':
            self.layernorms0 = torch.nn.ModuleList([torch.nn.LayerNorm(d_model, elementwise_affine=False, bias=False) for _ in range(n_layer)])
            self.layernorms1 = torch.nn.ModuleList([torch.nn.LayerNorm(d_model, elementwise_affine=False, bias=False) for _ in range(n_layer)])
        elif layernorm == 'Affine':
            self.layernorms0 = torch.nn.ModuleList([torch.nn.LayerNorm(d_model) for _ in range(n_layer)])
            self.layernorms1 = torch.nn.ModuleList([torch.nn.LayerNorm(d_model) for _ in range(n_layer)])

        if enorm == 'True':
            self.enorms = torch.nn.LayerNorm(d_model, elementwise_affine=False, bias=False)
        elif enorm == 'Affine':
            self.enorms = torch.nn.LayerNorm(d_model)
        self.enorm = enorm
        self.lmul = constant(1/ldiv)

    def forward(self, x: torch.Tensor, last_only: bool, capture_stats: bool) -> tuple[torch.Tensor, list]:
        x = torch.nn.functional.embedding(x.long(), self.embedding) + self.pos_embedding[:, :x.shape[1], :]
        if self.enorm != 'False':
            x = self.enorms(x) * self.lmul

        if capture_stats:
            stats = [['embed',x.norm().item()]]
        for layer in range(self.n_layer):
            if self.d_k > 0:
                # attention
                q = torch.einsum('btm,hmq->bhtq', x, self.wq[layer])
                k = torch.einsum('bom,hmk->bhok', x, self.wk[layer])
                v = torch.einsum('bom,hmv->bhov', x, self.wv[layer])
                attn = torch.einsum('bhtq,bhoq->bhto', q, k)
                attn = torch.exp(attn.clamp(max=50))
                attn = torch.tril(attn)
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)
                vsum = torch.einsum('bhto,bhov->btv', attn, v)
                x = x + vsum * self.amul
                if self.layernorm != 'False':
                    x = self.layernorms0[layer](x) * self.lmul
                if capture_stats:
                    stats.append([f'vsum{layer}',vsum.norm().item()])
                    stats.append([f'r_mid{layer}',x.norm().item()])

            # mlp
            y = torch.einsum('btm,mh->bth', x, self.mlp0[layer])
            y = torch.relu(y + self.mlpb[layer])
            y = torch.einsum('bth,hm->btm', y, self.mlp1[layer])
            x = x + y * self.pmul
            if self.layernorm != 'False':
                x = self.layernorms1[layer](x) * self.lmul
            if capture_stats:
                stats.append([f'y{layer}',y.norm().item()])
                stats.append([f'r_end{layer}',x.norm().item()])

        if last_only:
            x = x[:,-1:,:]
        result = torch.einsum('btm,md->btd', x, self.unembedding) + self.bias
        if capture_stats:
            return result, stats
        else:
            return result, None

def validation(model, batch, mask):
    with torch.no_grad():
        y, stats = model(batch,False,True)
        mr = mask[:,1:].reshape(-1)
        yr = y[:,:-1,:].reshape(-1, y.shape[-1])[mr]
        br = batch[:,1:].reshape(-1).long()[mr]
        #print(batch.shape, y.shape, mr.shape, yr.shape, br.shape)
        loss = torch.nn.functional.cross_entropy(yr, br, reduction='mean')
        #print(f'  Validation loss: {loss.item()}')
        #print(tokenizer.decode(batch[0]))
        return loss.item(), stats

def train(model, slurper, time_s, vbatch, vmask, device, tokenizer):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_sum = torch.zeros((), device=device)
    start_time = time.monotonic()
    results = []
    i = 0
    while time.monotonic() - start_time < time_s:
        batch, mask = slurper.batch()
        y,_ = model(batch,False,False)
        mr = mask[:,1:].reshape(-1)
        yr = y[:,:-1,:].reshape(-1, y.shape[-1])[mr]
        br = batch[:,1:].reshape(-1).long()[mr]
        loss = torch.nn.functional.cross_entropy(yr, br, reduction='mean')
        loss.backward()
        opt.step()
        opt.zero_grad()
        loss_sum += loss.detach()
        i += 1
        if i % 1000 == 0:
            #print(f'Batch {i+1}, loss: {loss_sum.item() / 1000}')
            loss_sum = 0
            #print(tokenizer.decode(batch[0]))
            vloss, stats = validation(model, vbatch, vmask)
            t = time.monotonic() - start_time
            pred = prediction_to_string(model, tokenizer, vbatch[0,:10])
            print(f'{t/60:8.3f} Batch {i}, loss: {vloss} {pred}')
            results.append({
                'time': t,
                'batch': i,
                'loss': vloss,
                'predictions': [pred],
                'stats': stats,
            })
    return results

def prediction_to_string(model, tokenizer, prompt_tokens, n_tokens=30):
    result = predict_slow(model, prompt_tokens, n_tokens, 0.8)
    return f'{tokenizer.decode(prompt_tokens).replace("\n","\\n")}-->{tokenizer.decode(result).replace("\n","\\n")}'

def predict_slow(model, prompt_tokens, n_tokens, temperature=0):
    result = torch.zeros(n_tokens, dtype=torch.uint16)
    prompt = prompt_tokens.reshape(1,-1)
    with torch.no_grad():
        for i in range(n_tokens):
            y,_ = model(prompt,True,False)
            if temperature == 0:
                next_token = torch.argmax(y[0,-1,:]).to(torch.uint16)
            else:
                probs = torch.softmax(y[0,-1,:] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).to(torch.uint16)
            prompt = torch.cat([prompt, next_token.reshape(1,1)], dim=1)
            result[i] = next_token
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('-o', type=str)
    parser.add_argument('--nlayer', type=int, default=1)
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--dmodel', type=int, default=64)
    parser.add_argument('--dk', type=int, default=4)
    parser.add_argument('--dhidden', type=int, default=128)
    parser.add_argument('--time', type=int, default=300)
    parser.add_argument('--mag', type=float, default=0.05)
    parser.add_argument('--adiv', type=float, default=1.0)
    parser.add_argument('--pdiv', type=float, default=1.0)
    parser.add_argument('--fixedpos', type=str, default='False')
    parser.add_argument('--layernorm', type=str, default='True')
    parser.add_argument('--enorm', type=str, default='False')
    parser.add_argument('--ldiv', type=float, default=1.0)
    args = parser.parse_args()
    input_filename = args.input_file
    dict_filename = f'{input_filename}.dictionary'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = Tokenizer(dict_filename)
    
    n_layer = args.nlayer
    n_head = args.nhead
    n_dict = len(tokenizer.tokens)
    n_batch = 1
    n_context = 128
    d_model = args.dmodel
    d_k = args.dk
    d_hidden = args.dhidden

    torch.manual_seed(12345)
    vslurper = DataSlurper(input_filename, 'validation', device, 64, n_context)
    slurper = DataSlurper(input_filename, 'train', device, n_batch, n_context)
    vbatch, vmask = vslurper.batch()

    model = TransformerModel(n_layer, n_head, n_dict, d_model, d_k, d_hidden, n_context, args.mag, args.adiv, args.pdiv, args.fixedpos, args.layernorm, args.enorm, args.ldiv).to(device)

    # with torch.no_grad():
    #     vbatch2 = torch.stack([vbatch[0]] * 2)
    #     vbatch2[1,5:] = 123
    #     print(vbatch2)
    #     y = model(vbatch2)
    #     print(y[:,:,0])

    print(f"Training with time={args.time} n_layer={n_layer}, n_head={n_head}, d_model={d_model}, d_k={d_k}, d_hidden={d_hidden}, mag={args.mag}, adiv={args.adiv}, pdiv={args.pdiv}, fixedpos={args.fixedpos}, layernorm={args.layernorm}, enorm={args.enorm}, ldiv={args.ldiv}")
    losses = train(model, slurper, args.time, vbatch, vmask, device, tokenizer)
    with open(args.o, 'w') as f:
        json.dump({
            'hyper': {
                'n_layer': n_layer,
                'n_head': n_head,
                'd_model': d_model,
                'd_k': d_k,
                'd_hidden': d_hidden,
                'mag': args.mag,
                'adiv': args.adiv,
                'pdiv': args.pdiv,
                'fixedpos': args.fixedpos,
                'layernorm': args.layernorm,
                'enorm': args.enorm,
                'ldiv': args.ldiv,
            },
            'losses': losses,
        }, f, indent=2)

if __name__ == '__main__':
    main()