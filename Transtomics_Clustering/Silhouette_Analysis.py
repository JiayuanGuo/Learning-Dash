import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('df5_log2_ratio.csv', index_col = ['locus_tag'])

app = dash.Dash()

app.layout = html.Div([
        dcc.Input(id='k-range', value= 30, type='number'),
    dcc.Graph(id='graph-silhouette_analysis')
    ])

@app.callback(
    dash.dependencies.Output('graph-silhouette_analysis', 'figure'),
    [dash.dependencies.Input(component_id='k-range',component_property='value')]
)

def silhouette_analysis(n):
    """
    n: the maximum of k value

    """
    k_values = np.array([])
    silhouette_scores = np.array([])

    K = range(2, n + 1)

    for i in K:
        # K-means for range of clusters
        kmeans = KMeans(n_clusters=i, max_iter=300, random_state=3)
        kmeans.fit(df)

        # Silhouette score for every k computed
        silhouette_ave = silhouette_score(df.values, kmeans.labels_)

        # x and y axis to plot k value and silhouette score
        k_values = np.append(k_values, [int(i)])
        silhouette_scores = np.append(silhouette_scores, [silhouette_ave])

    return {
        'data': [go.Scatter(
            x=k_values,
            y=silhouette_scores,
            mode= 'lines+markers',
            marker = dict(
                size='10')
        )],
        'layout': go.Layout(
            xaxis={'title': 'K Value'},
            yaxis={'title': 'Silhouette coefficient'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            hovermode='closest'
        )
    }

if __name__ == '__main__':
    app.run_server()
