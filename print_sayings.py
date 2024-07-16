import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    args = parser.parse_args()
    with open(args.input_file) as f:
        dataset = json.load(f)
        for item in dataset['losses']:
            print(f"{item['loss']:5.3f} {item['predictions'][0]}")

if __name__ == '__main__':
    main()