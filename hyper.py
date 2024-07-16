import argparse
import os
import subprocess

def run(input_file: str, n_layer=1, n_head=4, d_model=128, d_k=4, d_hidden=128, time_s=300, mag=0.125, adiv=10, pdiv=10, fixedpos='FromZero'):
    output_file = f'data/l{n_layer}_h{n_head}_d{d_model}_k{d_k}_h{d_hidden}_ad{adiv}_pd{pdiv}'
    if mag != 0.125:
        output_file += f'_m{mag}'
    if fixedpos == 'True':
        output_file += '_fixedpos'
    elif fixedpos == 'None':
        output_file += '_nopos'
    elif fixedpos == 'FromZero':
        output_file += '_fromzero'
    output_file += '.json'
    if os.path.exists(output_file):
        print(f'File {output_file} already exists')
    else:
        subprocess.run(['python', 'transformer2.py', input_file, '-o', output_file, '--time', str(time_s), '--nlayer', str(n_layer), '--nhead', str(n_head), '--dmodel', str(d_model), '--dk', str(d_k), '--dhidden', str(d_hidden), '--mag', str(mag), '--adiv', str(adiv), '--pdiv', str(pdiv), '--fixedpos', str(fixedpos)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    os.makedirs('data', exist_ok=True)
    inp = parser.parse_args().input_file
    run(inp, time_s=1200)
    run(inp, time_s=1200, n_layer=2)
    # run(inp, n_layer=1, fixedpos='True')
    # run(inp, n_layer=1)
    # run(inp, n_layer=1, fixedpos='None')
    # run(inp, n_layer=1, fixedpos='FromZero')
    # run(inp, n_layer=2, fixedpos='True')
    # run(inp, n_layer=2, fixedpos='FromZero')
    # run(inp, n_layer=2)
    #run(inp, n_layer=3, fixedpos='FromZero')
    #run(inp, n_layer=2)
    #run(inp, n_layer=1, mag=0.125)
    #run(inp, n_layer=1, adiv=10, pdiv=10, mag=0.125)
    # run(inp, n_layer=2, adiv=10)
    # run(inp, n_layer=2, pdiv=10)
    # run(inp, n_layer=2, adiv=10, pdiv=10)
    # run(inp, n_layer=2, adiv=10, pdiv=10, mag=0.3)
    # run(inp, n_layer=2, adiv=10, pdiv=10, mag=0.2)
    # run(inp, n_layer=2, adiv=10, pdiv=10, mag=0.125)
    # run(inp, n_layer=2, adiv=100, pdiv=100, mag=0.125)

if __name__ == '__main__':
    main()
