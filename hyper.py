import argparse
import os
import subprocess

def run(input_file: str, n_layer=1, n_head=4, d_model=128, d_k=4, d_hidden=128, time_s=300, mag=0.125, adiv=10, pdiv=10, fixedpos='FromZero', layernorm='False', enorm='False', ldiv=1, n_batch=1):
    output_file = f'data/l{n_layer}_h{n_head}_d{d_model}_k{d_k}_h{d_hidden}_ad{adiv}_pd{pdiv}'
    if mag != 0.125:
        output_file += f'_m{mag}'
    if fixedpos == 'True':
        output_file += '_fixedpos'
    elif fixedpos == 'None':
        output_file += '_nopos'
    elif fixedpos == 'FromZero':
        output_file += '_fromzero'
    if layernorm == 'True':
        output_file += '_layernorm'
    elif layernorm == 'Affine':
        output_file += '_affine'
    if enorm == 'True':
        output_file += '_enorm'
    elif enorm == 'Affine':
        output_file += '_eaffine'
    if ldiv != 1:
        output_file += f'_ld{ldiv}'
    if n_batch != 1:
        output_file += f'_b{n_batch}'
    output_file += '.json'
    if os.path.exists(output_file):
        print(f'File {output_file} already exists')
    else:
        subprocess.run(['python', 'transformer2.py', input_file, '-o', output_file, '--time', str(time_s), '--nlayer', str(n_layer), '--nhead', str(n_head), '--dmodel', str(d_model), '--dk', str(d_k), '--dhidden', str(d_hidden), '--mag', str(mag), '--adiv', str(adiv), '--pdiv', str(pdiv), '--fixedpos', str(fixedpos), '--layernorm', str(layernorm), '--enorm', str(enorm), '--ldiv', str(ldiv), '--batch', str(n_batch)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    os.makedirs('data', exist_ok=True)
    inp = parser.parse_args().input_file
    # run(inp, layernorm='Affine', enorm='Affine')
    # run(inp, n_layer=2, layernorm='Affine', enorm='Affine')
    # run(inp, layernorm='Affine', enorm='Affine', adiv=30, pdiv=30)
    # run(inp, layernorm='Affine', enorm='Affine', adiv=3, pdiv=3)
    # run(inp, layernorm='Affine', ldiv=3)
    # run(inp, n_layer=2, layernorm='Affine', ldiv=3)
    # run(inp, n_layer=2, layernorm='Affine', ldiv=10)
    run(inp, n_layer=2, layernorm='Affine', ldiv=3, n_batch=2)
    run(inp, n_layer=2, layernorm='Affine', ldiv=3, n_batch=4)
    run(inp, n_layer=2, layernorm='Affine', ldiv=3, n_batch=8)
    run(inp, n_layer=2, layernorm='Affine', ldiv=3, n_batch=6)
    run(inp, n_layer=3, layernorm='Affine', ldiv=3, n_batch=4)

if __name__ == '__main__':
    main()
