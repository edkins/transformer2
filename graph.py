import argparse
import json
import os
import plotly.graph_objects as go

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', type=str, choices=['time','batch'])
    args = parser.parse_args()
    with os.scandir('data') as it:
        data = []
        for entry in it:
            if entry.name.endswith('.json') and entry.is_file():
                with open(entry.path) as f:
                    item = json.load(f)
                    item['filename'] = entry.name
                    data.append(item)
    
    fig = go.Figure()
    for dataset in data:
        losses = dataset['losses']
        if args.kind == 'batch':
            xs = [point['batch'] for point in losses]
        else:
            xs = [point['time'] for point in losses]
        ys = [point['loss'] for point in losses]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=dataset['filename']))
    fig.update_layout(xaxis_title=args.kind, yaxis_title='loss')
    fig.show()

if __name__ == '__main__':
    main()