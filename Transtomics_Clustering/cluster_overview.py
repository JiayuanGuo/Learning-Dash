import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv('df5_log2_ratio.csv', index_col = ['locus_tag'])

app = dash.Dash()

app.layout = html.Div([
        dcc.Input(id='my-id', value='15', type='number'),
    dcc.Graph(id='graph-with-slider')
    ])


@app.callback(
    dash.dependencies.Output('graph-with-slider', 'figure'),
    [dash.dependencies.Input(component_id='my-id',component_property='value')]
)

def cluster_size_figure(kvalue):
    X = df

    kmeans = KMeans(n_clusters=kvalue, max_iter=300, random_state=4)
    kmeans.fit(X)

    labels_kmeans = kmeans.labels_
    df_clusterid = pd.DataFrame(labels_kmeans, index=df.index)
    df_clusterid.rename(columns={0: "cluster"}, inplace=True)
    df_clusters = pd.concat([df, df_clusterid], axis=1)

    count = df_clusters.groupby('cluster').count().iloc[:, 0]

    y_stdev = df_clusters.groupby("cluster").std()
    y_mean = df_clusters.groupby("cluster").mean()

    y_low = y_mean.subtract(y_stdev, fill_value=0)
    y_high = y_mean.add(y_stdev, fill_value=0)

    return {
        'data':[go.Bar(
            x = list(count.index),
            y = count.values
            #mode='markers',
            #opacity=0.7
            #marker={
                #'size': 15,
                #'line': {'width': 0.5, 'color': 'white'}
            #},
        )],
        'layout': go.Layout(
            xaxis = {'title':'cluster id'},
            yaxis = {'title':'number of genes in each cluster'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            hovermode='closest'
        )
    }

if __name__ == '__main__':
    app.run_server()
