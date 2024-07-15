import argparse
import os
import subprocess

def run(input_file: str, n_layer=1, n_head=4, d_model=128, d_k=4, d_hidden=128, time_s=300, mag=0.1):
    output_file = f'data/l{n_layer}_h{n_head}_d{d_model}_k{d_k}_h{d_hidden}'
    if mag != 0.05 or True:
        output_file += f'_m{mag}'
    output_file += '.json'
    if os.path.exists(output_file):
        print(f'File {output_file} already exists')
    else:
        subprocess.run(['python', 'transformer2.py', input_file, '-o', output_file, '--time', str(time_s), '--nlayer', str(n_layer), '--nhead', str(n_head), '--dmodel', str(d_model), '--dk', str(d_k), '--dhidden', str(d_hidden), '--mag', str(mag)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    os.makedirs('data', exist_ok=True)
    inp = parser.parse_args().input_file
    run(inp, n_layer=2, mag=0.03)
    run(inp, n_layer=2, mag=0.05)
    run(inp, n_layer=2, mag=0.1)
    run(inp, n_layer=2, mag=0.2)

if __name__ == '__main__':
    main()
