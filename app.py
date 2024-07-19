import torch
from transformer2 import TransformerModel, Tokenizer, predict_slow
from flask import Flask, make_response, request
import os

model = {}
tokenizer = {}
hyper = {}

def load(name: str):
    global model
    global tokenizer
    global hyper

    if name in model:
        return model[name], tokenizer[name], hyper[name]

    model_file = f'data/{name}.model'
    print(f'Loading model {model_file}')
    with open(model_file, 'rb') as f:
        things = torch.load(f)
        h = things['hyper']
        hyper[name] = h
        n_context = h['n_context']
        model[name] = TransformerModel(
            n_layer = h['n_layer'],
            n_head = h['n_head'],
            n_dict = h['n_dict'],
            d_model = h['d_model'],
            d_k = h['d_k'],
            d_hidden = h['d_hidden'],
            n_context = n_context,
            mag = h['mag'],
            adiv = h['adiv'],
            pdiv = h['pdiv'],
            fixedpos = h['fixedpos'],
            layernorm = h['layernorm'],
            enorm = h['enorm'],
            ldiv = h['ldiv'],
            vsmall = h['vsmall'],
        )
        model[name].load_state_dict(things['model'])

    tokenizer[name] = Tokenizer(tokens=things['dictionary'])
    return model[name], tokenizer[name], hyper[name]

app = Flask(__name__)

@app.get('/')
def index():
    return app.redirect('/static/index.html')

@app.get('/api/models')
def models():
    with os.scandir('data') as it:
        return {
            'models': list(sorted(entry.name[:-len('.model')] for entry in it if entry.is_file() and entry.name.endswith('.model')))
        }

@app.post('/api/tokenize')
def tokenize():
    model, tokenizer, hyper = load(request.json['model'])
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
    model, tokenizer, hyper = load(request.json['model'])
    return hyper

@app.post('/api/attention')
def attention():
    model, tokenizer, hyper = load(request.json['model'])
    tokens = tokenizer.encode_slow(request.json['prompt'])
    _, attns = model(tokens.reshape(1,-1), capture='attn')
    attn_bytes = attns.to(torch.float32).reshape(-1).detach().cpu().numpy().tobytes()
    return make_response(attn_bytes, 200, {
        'Content-Type': 'application/octet-stream',
        'X-heads': str(hyper['n_head']),
        'X-layers': str(hyper['n_layer']),
        'X-tokens': str(len(tokens)),
    })

@app.post('/api/predict')
def probs():
    model, tokenizer, hyper = load(request.json['model'])
    tokens = tokenizer.encode_slow(request.json['prompt'])
    logits = model(tokens.reshape(1,-1), last_only=True)[0].reshape(-1)
    probs = torch.softmax(logits, dim=-1)
    return {
        'next': list(sorted((
            {
                'token': i,
                'name': tokenizer.lookup_name(i),
                'prob': prob.item(),
            }
            for i, prob in enumerate(probs)), key=lambda x: -x['prob'])
        )
    }
