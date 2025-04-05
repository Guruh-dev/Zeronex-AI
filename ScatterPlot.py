import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Inisialisasi aplikasi Dash dengan tema dark
app = Dash(_name_, external_stylesheets=[dbc.themes.DARKLY])

# Parameter data sintetis
num_clusters = 3
num_samples = 200
num_time = 10  # jumlah frame waktu (misal 10 time steps)

# Definisi cluster: mean awal, pergeseran (shift) tiap waktu, dan matriks kovarians
initial_means = np.array([[2, 2], [7, 7], [12, 2]])
shifts = np.array([[0.5, 0.2], [-0.3, 0.4], [0.2, -0.5]])
covs = [
    [[1, 0.5], [0.5, 1]],
    [[1, -0.3], [-0.3, 1]],
    [[1, 0.2], [0.2, 1]]
]

# Membuat data yang bergantung pada waktu untuk setiap cluster
data = []
for t in range(num_time):
    for i in range(num_clusters):
        # Update mean cluster berdasarkan waktu (linear shift)
        mean = initial_means[i] + t * shifts[i]
        cluster_data = np.random.multivariate_normal(mean, covs[i], num_samples)
        for point in cluster_data:
            data.append({
                'time': t,
                'X': point[0],
                'Y': point[1],
                'Cluster': f'Cluster {i+1}'
            })

# Konversi data ke DataFrame
df = pd.DataFrame(data)

# Buat scatter plot interaktif dengan animasi menggunakan Plotly Express
fig = px.scatter(
    df,
    x='X',
    y='Y',
    color='Cluster',
    animation_frame='time',
    title="Interactive Animated Scatter Plot - Advanced Dashboard",
    hover_data=['Cluster']
)

# Kustomisasi tampilan marker dan layout
fig.update_traces(marker=dict(
    size=8,
    line=dict(width=1, color='DarkSlateGrey')
))
fig.update_layout(
    template="plotly_dark",
    xaxis_title="Sumbu X",
    yaxis_title="Sumbu Y",
    legend_title="Cluster",
    title_font=dict(size=24, family='Arial', color='white')
)

# Layout aplikasi dengan Dash dan Bootstrap untuk tampilan responsif
app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(html.H1("Advanced Zeronex AI Dashboard", className="text-center text-white mb-4"), width=12)
    ),
    dbc.Row(
        dbc.Col(
            dcc.Graph(id="animated-scatter", figure=fig),
            width=12
        )
    ),
    dbc.Row(
        dbc.Col(
            dcc.Slider(
                id='time-slider',
                min=0,
                max=num_time - 1,
                value=0,
                marks={i: str(i) for i in range(num_time)},
                step=None
            ),
            width=12,
            className="mt-4"
        )
    )
], fluid=True)

# Callback untuk update scatter plot berdasarkan input slider waktu
@app.callback(
    Output("animated-scatter", "figure"),
    Input("time-slider", "value")
)
def update_figure(selected_time):
    # Filter DataFrame untuk waktu yang dipilih
    filtered_df = df[df['time'] == selected_time]
    updated_fig = px.scatter(
        filtered_df,
        x='X',
        y='Y',
        color='Cluster',
        title=f"Scatter Plot pada Waktu {selected_time}",
        hover_data=['Cluster']
    )
    updated_fig.update_traces(marker=dict(
        size=10,
        line=dict(width=1, color='DarkSlateGrey')
    ))
    updated_fig.update_layout(
        template="plotly_dark",
        xaxis_title="Sumbu X",
        yaxis_title="Sumbu Y",
        legend_title="Cluster",
        title_font=dict(size=24, family='Arial', color='white')
    )
    return updated_fig

if __name__ == '_main_':
    app.run_server(debug=True)