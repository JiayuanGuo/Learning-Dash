import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist, pdist

df = pd.read_csv('df5_log2_ratio.csv', index_col = ['locus_tag'])

#Elbow Method

app = dash.Dash()

app.layout = html.Div([
        dcc.Input(id='k-range', value= 30, type='number'),
    #dcc.Graph(id='graph-cluster-profile'),
    dcc.Graph(id='graph-elbow_method')
    ])

@app.callback(
    dash.dependencies.Output('graph-elbow_method', 'figure'),
    [dash.dependencies.Input(component_id='k-range',component_property='value')]
)
def elbow_method_evaluation(n):
    """
    n: the maximum of k value

    """
    # Fit the kmeans model for k in a certain range
    K = range(1, n + 1)
    KM = [KMeans(n_clusters=k).fit(df) for k in K]
    # Pull out the cluster centroid for each model
    centroids = [k.cluster_centers_ for k in KM]

    # Calculate the distance between each data point and the centroid of its cluster
    k_euclid = [cdist(df.values, cent, 'euclidean') for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]

    # Total within sum of square
    wss = [sum(d ** 2) / 1000 for d in dist]
    # The total sum of square
    tss = sum(pdist(df.values) ** 2) / df.values.shape[0]
    # The between-clusters sum of square
    bss = tss - wss

    return {
        'data': [go.Bar(
            x=list(K),
            y=list(wss)
        )],
        'layout': go.Layout(
            xaxis={'title': 'K Value'},
            yaxis={'title': 'Sum of Within-cluster Distance/1000'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            hovermode='closest'
        )
    }


if __name__ == '__main__':
    app.run_server()
