import argparse
import json
import plotly.graph_objects as go

def get_stat(stats: list, stat_name: str):
    for name,value in stats:
        if name == stat_name:
            return value
    raise ValueError(f'Stat {stat_name} not found')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    args = parser.parse_args()
    with open(args.input_file) as f:
        dataset = json.load(f)
        dataset['filename'] = args.input_file

    fig = go.Figure()
    losses = dataset['losses']
    xs = [point['time'] for point in losses]
    stat_names = [name for name,_ in dataset['losses'][0]['stats']]
    for stat in stat_names:
        ys = [get_stat(point['stats'], stat) for point in dataset['losses']]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=stat))
        fig.update_layout(xaxis_title='time', yaxis_title='stat')
    fig.show()

if __name__ == '__main__':
    main()