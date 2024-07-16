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
    
    data.sort(key=lambda x: (x['hyper']['n_layer'], x['hyper']['n_head'], x['hyper']['d_model'], x['hyper']['d_k'], x['hyper']['d_hidden'], x['hyper'].get('adiv',1), x['hyper'].get('pdiv',1), x['filename']))

    #colors = ['#800','#f00','#f88','#fcc', '#004','#00f','#44f','#ccf','#080']

    fig = go.Figure()
    for i, dataset in enumerate(data):
        losses = dataset['losses']
        if args.kind == 'batch':
            xs = [point['batch'] for point in losses]
            xlabel = 'batch'
        else:
            xs = [point['time']/60 for point in losses]
            xlabel = 'time (minutes)'
        ys = [point['loss'] for point in losses]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=dataset['filename'])) #, line=dict(color=colors[i])))
    fig.update_layout(xaxis_title=xlabel, yaxis_title='loss')
    fig.show()

if __name__ == '__main__':
    main()