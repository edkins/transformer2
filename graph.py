import argparse
import json
import os
import plotly.graph_objects as go

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', type=str, choices=['time','data','ratio','ratiodata','tloss'])
    args = parser.parse_args()
    with os.scandir('data') as it:
        data = []
        for entry in it:
            if entry.name.endswith('.json') and entry.is_file():
                with open(entry.path) as f:
                    item = json.load(f)
                    item['filename'] = entry.name
                    data.append(item)
    
    data.sort(key=lambda x: (x['hyper']['n_layer'], x['hyper']['n_head'], x['hyper']['d_model'], x['hyper']['d_k'], x['hyper']['d_hidden'], x['hyper'].get('adiv',1), x['hyper'].get('pdiv',1), x['filename']))

    #colors = ['#800','#f00','#f88','#fcc', '#004','#00f','#44f','#ccf','#080']

    fig = go.Figure()
    if len(data) == 0:
        raise Exception('No data found')

    for i, dataset in enumerate(data):
        losses = dataset['losses']
        if args.kind in ['data','ratiodata']:
            xs = [point['batch'] for point in losses]
            xlabel = 'datapoint'
        else:
            xs = [point['time']/60 for point in losses]
            xlabel = 'time (minutes)'

        if args.kind in ['ratio','ratiodata']:
            if 'ratio' not in losses[0]:
                continue
            ys = [point['ratio'] for point in losses]
            yaxis_title = 'compression ratio'
        elif args.kind == 'tloss':
            ys = [point['tloss'] for point in losses]
            yaxis_title = 'training loss'
        else:
            ys = [point['loss'] for point in losses]
            yaxis_title = 'loss'
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=dataset['filename'])) #, line=dict(color=colors[i])))
    fig.update_layout(xaxis_title=xlabel, yaxis_title=yaxis_title)
    fig.show()

if __name__ == '__main__':
    main()