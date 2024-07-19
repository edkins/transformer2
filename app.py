import torch
from transformer2 import TransformerModel, Tokenizer, predict_slow
from flask import Flask, request
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
