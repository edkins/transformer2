import argparse
import base64
import json
import math
import sys
import time
import torch
from typing import Literal

class StdinSlurper:
    def __init__(self, device: str, n_batch: int, n_context: int):
        self.device = device
        self.n_batch = n_batch
        self.n_context = n_context

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns an int16 tensor of shape (n, length) read from stdin as binary.
        """
        with torch.no_grad():
            read_bytes = sys.stdin.buffer.read(self.n_batch * self.n_context * 2)
            batch = torch.frombuffer(read_bytes, dtype=torch.int16).reshape(self.n_batch, self.n_context).to(self.device)
            mask = batch != 0
            mask[0,:] = True
            return batch, mask

class StdoutWriter:
    def __init__(self):
        pass

    def write(self, tensor: torch.Tensor):
        sys.stdout.buffer.write(tensor.cpu().numpy().tobytes())

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
        Returns an int16 tensor of shape (n, length) containing the first `length` tokens of `n` random articles from the specified split.
        Each row will start with a zero byte which is the beginning-of-sequence token.

        Also returns a bool tensor of shape (n, length) containing the part of the tensor that is actually filled with data.
        (The data may be padded with zeros in the case that the end of the article is reached).
        """
        with torch.no_grad():
            byte_length = 2 * (self.n_context - 1)
            articles = self._pick_articles(self.n_batch)
            result = torch.zeros((self.n_batch, self.n_context), dtype=torch.int16)
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
    
    def encode_slow(self, string: str, prepend_sep=True) -> torch.Tensor:
        string = string.encode('utf-8')
        remaining = string
        result = []
        if prepend_sep:
            result = [0]
        while remaining != b'':
            best = b''
            best_i = -1
            for i,token in enumerate(self.tokens):
                if i == 0:
                    continue
                if remaining.startswith(token) and len(token) > len(best):
                    best = token
                    best_i = i
            result.append(best_i)
            remaining = remaining[len(best):]
        return torch.tensor(result, dtype=torch.int16)

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
        self.n_dict = n_dict
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

def validation(model, vdata, vmask):
    with torch.no_grad():
        loss = torch.zeros((), device=vdata.device)
        count = 0
        for i in range(0, len(vdata), 4):
            batch = vdata[i:i+4]
            mask = vmask[i:i+4]
            y, stats = model(batch,False,True)
            mr = mask[:,1:].reshape(-1)
            yr = y[:,:-1,:].reshape(-1, y.shape[-1])[mr]
            br = batch[:,1:].reshape(-1).long()[mr]
            #print(batch.shape, y.shape, mr.shape, yr.shape, br.shape)
            loss += torch.nn.functional.cross_entropy(yr, br, reduction='mean')
            #print(f'  Validation loss: {loss.item()}')
            #print(tokenizer.decode(batch[0]))
            count += 1
        return loss.item() / count, stats

def train(model, slurper, time_s, vbatch, vmask, device, tokenizer, vcompress, vcmask, vcbits):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_sum = torch.zeros((), device=device)
    start_time = time.monotonic()
    results = []
    i = 0
    target_i = 1000
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
        i += len(batch)
        if i >= target_i:
            #print(f'Batch {i+1}, loss: {loss_sum.item() / 1000}')
            loss_sum = 0
            #print(tokenizer.decode(batch[0]))
            vloss, stats = validation(model, vbatch, vmask)
            vratio = compression_ratio(model, vcompress, vcmask, vcbits)
            t = time.monotonic() - start_time
            pred = prediction_to_string(model, tokenizer, vbatch[0,:10])
            print(f'{t/60:8.3f} {i:7d}, loss: {vloss:6.4f} ratio: {vratio:6.4f} {pred}')
            results.append({
                'time': t,
                'batch': i,
                'loss': vloss,
                'ratio': vratio,
                'predictions': [pred],
                'stats': stats,
            })
            target_i += 1000
    return results

def compression_ratio(model: TransformerModel, vbatch_data: torch.Tensor, vmask_data: torch.Tensor, vbits: int) -> float:
    """
    Find the negative base2-log likelihood of the model outputting vbatch (where some of the entries are irrelevant -
    those are masked out by vmask).

    Then divide by the number of bits in the target, to get a compression ratio.
    """
    with torch.no_grad():
        size = 0
        for i in range(0, len(vbatch_data), 4):
            vbatch = vbatch_data[i:i+4]
            vmask = vmask_data[i:i+4]
            y,_ = model(vbatch,False,False)
            mr = vmask[:,1:].reshape(-1)
            logprobs = torch.nn.functional.log_softmax(y, dim=-1) / math.log(2)
            #lpr = torch.nn.functional.embedding(vbatch.long(), logprobs)
            lpr = logprobs[:,:-1,:].reshape(-1, logprobs.shape[-1])[mr]
            br = vbatch[:,1:].reshape(-1).long()[mr]
            width = lpr.shape[0]
            size += -torch.sum(lpr[torch.arange(0, width), br])
        return (size / vbits).item()

def prediction_to_string(model, tokenizer, prompt_tokens, n_tokens=30, temperature=0.8):
    result = predict_slow(model, prompt_tokens, n_tokens, temperature)
    left = tokenizer.decode(prompt_tokens).replace("\n","\\n")
    right = tokenizer.decode(result).replace("\n","\\n")
    return f'{left}-->{right}'

def predict_slow(model, prompt_tokens, n_tokens, temperature=0):
    result = torch.zeros(n_tokens, dtype=torch.int16)
    prompt = prompt_tokens.reshape(1,-1)
    with torch.no_grad():
        for i in range(n_tokens):
            y,_ = model(prompt,True,False)
            if temperature == 0:
                next_token = torch.argmax(y[0,-1,:]).to(torch.int16)
            else:
                probs = torch.softmax(y[0,-1,:] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).to(torch.int16)
            prompt = torch.cat([prompt, next_token.reshape(1,1)], dim=1)
            result[i] = next_token
        return result

def final_predictions(model: TransformerModel, tokenizer: Tokenizer, batch: torch.Tensor) -> list[str]:
    with torch.no_grad():
        result = []
        for i in range(len(batch)):
            prompt = batch[i,:10]
            for temperature in [0, 0.25, 0.5, 0.8, 1.0]:
                string = prediction_to_string(model, tokenizer, prompt)
                result.append(f'{temperature:5.3f} {string}')
        return result

def gen_compress(vbatch, vmask, tokenizer, filename_base) -> list[str]:
    tokenlen = vbatch.shape[1]//2
    strings = []
    with torch.no_grad():
        for i in range(len(vbatch)):
            tlen = min(tokenlen, vmask[i,:].sum()-1)
            string = tokenizer.decode(vbatch[i,1:tlen])
            if len(string) > 25:
                strings.append(string)
    with open(f'{filename_base}.validation.compress','w') as f:
        json.dump(strings, f, indent=2)

def load_compress(filename_base: str, tokenizer: Tokenizer, width: int, device: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    with open(f'{filename_base}.validation.compress') as f:
        strings = json.load(f)
        return tokenize_compress(strings, tokenizer, width, device)

def tokenize_compress(strings: list[str], tokenizer: Tokenizer, width: int, device: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    result = torch.zeros((len(strings), width), dtype=torch.int16)
    rmask = torch.zeros((len(strings), width), dtype=bool)
    rbits = 0
    for i,string in enumerate(strings):
        toks = tokenizer.encode_slow(string)
        if len(toks) > width:
            raise Exception("tokenize_compress: width is not enough")
        result[i, :len(toks)] = toks
        rmask[i, :len(toks)] = True
        rbits += len(string) * 8
    return result.to(device), rmask.to(device), rbits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str) # options=['slurp-out', 'slurp-in', 'train']
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
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--gencompress', action='store_true')
    args = parser.parse_args()
    input_filename = args.input_file
    dict_filename = f'{input_filename}.dictionary'
    if torch.cuda.is_available() and args.command in ['slurp-in', 'train']:
        device = 'cuda'
    else:
        device = 'cpu'
    tokenizer = Tokenizer(dict_filename)
    
    n_layer = args.nlayer
    n_head = args.nhead
    n_dict = len(tokenizer.tokens)
    n_batch = args.batch
    n_context = 128
    d_model = args.dmodel
    d_k = args.dk
    d_hidden = args.dhidden

    torch.manual_seed(12345)
    if args.command in ['slurp-in', 'train']:
        vslurper = DataSlurper(input_filename, 'validation', device, 64, n_context)
        vbatch, vmask = vslurper.batch()
        if args.gencompress:
            gen_compress(vbatch, vmask, tokenizer, input_filename)
        vcompress, vcmask, vbits = load_compress(input_filename, tokenizer, n_context, device)
        model = TransformerModel(n_layer, n_head, n_dict, d_model, d_k, d_hidden, n_context, args.mag, args.adiv, args.pdiv, args.fixedpos, args.layernorm, args.enorm, args.ldiv).to(device)

    if args.command in ['slurp-out', 'train']:
        slurper = DataSlurper(input_filename, 'train', device, n_batch, n_context)
    elif args.command == 'slurp-in':
        slurper = StdinSlurper(device, n_batch, n_context)


    if args.command == 'slurp-out':
        writer = StdoutWriter()
        i = 0
        target_i = 10_000
        while True:
            batch, mask = slurper.batch()
            i += len(batch)
            writer.write(batch)
            writer.write(mask)
            if i >= target_i:
                sys.stderr.write(f'Sent datapoint {i}\n')
                target_i += 10_000
        # doesn't ever finish. Just carries on slurping.

    # with torch.no_grad():
    #     vbatch2 = torch.stack([vbatch[0]] * 2)
    #     vbatch2[1,5:] = 123
    #     print(vbatch2)
    #     y = model(vbatch2)
    #     print(y[:,:,0])

    print(f"Training with time={args.time} n_layer={n_layer}, n_head={n_head}, n_batch={n_batch}, d_model={d_model}, d_k={d_k}, d_hidden={d_hidden}, mag={args.mag}, adiv={args.adiv}, pdiv={args.pdiv}, fixedpos={args.fixedpos}, layernorm={args.layernorm}, enorm={args.enorm}, ldiv={args.ldiv}")
    losses = train(model, slurper, args.time, vbatch, vmask, device, tokenizer, vcompress, vcmask, vbits)
    predictions = final_predictions(model, tokenizer, vbatch)
    with open(args.o, 'w') as f:
        json.dump({
            'hyper': {
                'n_layer': n_layer,
                'n_head': n_head,
                'd_model': d_model,
                'd_k': d_k,
                'd_hidden': d_hidden,
                'n_dict': n_dict,
                'mag': args.mag,
                'adiv': args.adiv,
                'pdiv': args.pdiv,
                'fixedpos': args.fixedpos,
                'layernorm': args.layernorm,
                'enorm': args.enorm,
                'ldiv': args.ldiv,
            },
            'losses': losses,
            'final_predictions': predictions,
        }, f, indent=2)

if __name__ == '__main__':
    main()