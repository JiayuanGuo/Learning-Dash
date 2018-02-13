import pandas as pd

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from sklearn.cluster import KMeans

df = pd.read_csv('df5_log2_ratio.csv', index_col = ['locus_tag'])

app = dash.Dash()

app.layout = html.Div([
    html.Div([
        dcc.Input(id='my-id', value='gene name', type='text'),
        dcc.Slider(
            id='k-value--slider',
            min=11,
            max=30,
            step=1,
            value=15,
            marks={i: '{}'.format(i) for i in range(11,30)},
        )
    ]),
    html.Div(id='my-div')

])


@app.callback(
    Output(component_id='my-div',component_property='children'),
    [Input(component_id='my-id',component_property='value'),
     Input('k-value--slider', 'value')]
)
def update_output_div(input_gene, k_value):

    X = df

    kmeans = KMeans(n_clusters=k_value, max_iter=300, random_state=4)
    kmeans.fit(X)

    labels_kmeans = kmeans.labels_
    df_clusterid = pd.DataFrame(labels_kmeans, index=df.index)
    df_clusterid.rename(columns={0: "cluster"}, inplace=True)
    #df_clusters = pd.concat([df, df_clusterid], axis=1)

    genes_clusterid = df_clusterid.loc[input_gene]

    return genes_clusterid


if __name__ == '__main__':
    app.run_server()
