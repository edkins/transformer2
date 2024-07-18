import argparse
import os
import pathlib
import subprocess

def run(input_file: str, mode='mem', n_layer=2, n_head=4, d_model=128, d_k=4, d_hidden=128, time_s=300, mag=0.125, adiv=10, pdiv=10, fixedpos='FromZero', layernorm='Affine', enorm='False', ldiv=3, n_batch=64, gamma=0, ratiolr=False, lr=0.01, epoch=10000):
    output_file = f'data/'
    if input_file.endswith('t4k'):
        output_file += '4k_'
    elif input_file.endswith('t8k'):
        output_file += '8k_'
    elif input_file.endswith('t12k'):
        output_file += '12k_'
    elif input_file.endswith('tokenized') or input_file.endswith('t16k'):
        output_file += '16k_'
    elif input_file.endswith('t24k'):
        output_file += '24k_'
    elif input_file.endswith('t32k'):
        output_file += '32k_'
    output_file += f'l{n_layer}_h{n_head}_d{d_model}_k{d_k}_h{d_hidden}_ad{adiv}_pd{pdiv}'
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
    if gamma != 0:
        output_file += f'_g{gamma}'
    if lr != 0.01:
        output_file += f'_lr{lr}'
    if ratiolr:
        output_file += '_ratiolr'
    output_file += '.json'
    if os.path.exists(output_file):
        print(f'File {output_file} already exists')
    else:
        options = [input_file, '-o', output_file, '--time', str(time_s), '--nlayer', str(n_layer), '--nhead', str(n_head), '--dmodel', str(d_model), '--dk', str(d_k), '--dhidden', str(d_hidden), '--mag', str(mag), '--adiv', str(adiv), '--pdiv', str(pdiv), '--fixedpos', str(fixedpos), '--layernorm', str(layernorm), '--enorm', str(enorm), '--ldiv', str(ldiv), '--batch', str(n_batch), '--gamma', str(gamma), '--ratiolr', str(ratiolr), '--lr', str(lr), '--epoch', str(epoch)]
        #ps = subprocess.Popen(['python', 'transformer2.py', 'slurp-out', *options], stdout=subprocess.PIPE)
        #subprocess.run(['python', 'transformer2.py', 'slurp-in', *options], stdin=ps.stdout)
        #ps.wait()
        subprocess.run(['python', 'transformer2.py', mode, *options])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str)
    os.makedirs('data', exist_ok=True)
    inp0 = str(pathlib.Path(parser.parse_args().directory, 'tokenized'))
    inp1 = str(pathlib.Path(parser.parse_args().directory, 't32k'))
    inp2 = str(pathlib.Path(parser.parse_args().directory, 't24k'))
    print(inp0)
    #print(inp1)
    #print(inp2)
    #run(inp0, time_s=30, mode='train', gamma=1, n_batch=4, epoch=1000)
    run(inp0, time_s=300, mode='mem', gamma=1)

if __name__ == '__main__':
    main()
