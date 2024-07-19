import torch
from transformer2 import TransformerModel, Tokenizer, predict_slow
from flask import Flask, make_response, request
import os

model = None
tokenizer = None
hyper = None

def load():
    global model
    global tokenizer
    global hyper

    if model is not None:
        return model, tokenizer, hyper

    model_file = os.environ['MODEL']
    print(f'Loading model {model_file}')
    with open(model_file, 'rb') as f:
        things = torch.load(f)
        hyper = things['hyper']
        n_context = hyper['n_context']
        model = TransformerModel(
            n_layer = hyper['n_layer'],
            n_head = hyper['n_head'],
            n_dict = hyper['n_dict'],
            d_model = hyper['d_model'],
            d_k = hyper['d_k'],
            d_hidden = hyper['d_hidden'],
            n_context = n_context,
            mag = hyper['mag'],
            adiv = hyper['adiv'],
            pdiv = hyper['pdiv'],
            fixedpos = hyper['fixedpos'],
            layernorm = hyper['layernorm'],
            enorm = hyper['enorm'],
            ldiv = hyper['ldiv'],
            vsmall = hyper['vsmall'],
        )
        model.load_state_dict(things['model'])

    tokenizer = Tokenizer(tokens=things['dictionary'])
    return model, tokenizer, hyper

app = Flask(__name__)
load()

@app.get('/')
def index():
    return app.redirect('/static/index.html')

@app.post('/api/tokenize')
def tokenize():
    model, tokenizer, hyper = load()
    tokens = tokenizer.encode_slow(request.json['prompt'])
    return {
        'tokens': [
            {
                'token': token.item(),
                'name': tokenizer.lookup_name(token.item()),
            }
            for token in tokens
        ]
    }

@app.post('/api/hyper')
def hyperparams():
    model, tokenizer, hyper = load()
    return hyper

@app.post('/api/attention')
def attention():
    model, tokenizer, hyper = load()
    tokens = tokenizer.encode_slow(request.json['prompt'])
    _, attns = model(tokens.reshape(1,-1), capture='attn')
    attn_bytes = attns.to(torch.float32).reshape(-1).detach().cpu().numpy().tobytes()
    return make_response(attn_bytes, 200, {
        'Content-Type': 'application/octet-stream',
        'X-heads': str(hyper['n_head']),
        'X-layers': str(hyper['n_layer']),
        'X-tokens': str(len(tokens)),
    })
