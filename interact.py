import argparse
import readline
import torch
from transformer2 import TransformerModel, Tokenizer, predict_slow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('temperature', type=float)
    args = parser.parse_args()
    print(f'Loading model {args.model}')
    with open(args.model, 'rb') as f:
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
    while True:
        prompt = input('> ')
        prompt_tokens = tokenizer.encode_slow(prompt)
        tokens = predict_slow(model, prompt_tokens, n_context - len(prompt_tokens), temperature=args.temperature)
        print(tokenizer.decode(tokens))

if __name__ == '__main__':
    main()