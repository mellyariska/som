# ============================
# SOM Clustering for Rainfall Data (NetCDF)
# ============================

import xarray as xr
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ============================
# 1. BACA DATA NETCDF
# ============================
file_path = "monthly_rr_0.25deg_reg_v2.0_saobs.nc"  # Ganti sesuai lokasi file kamu
ds = xr.open_dataset(file_path)
print("Dataset loaded. Time range:", str(ds['time'].values[0]), "to", str(ds['time'].values[-1]))

# Hitung rata-rata bulanan (climatology Jan–Des)
monthly_clim = ds['rr'].groupby('time.month').mean(dim='time')  # (month, lat, lon)
print("Monthly climatology shape:", monthly_clim.shape)

lat_vals = ds['latitude'].values
lon_vals = ds['longitude'].values
nlat = len(lat_vals)
nlon = len(lon_vals)

# ============================
# 2. SUSUN FITUR UNTUK SOM
# ============================
clim_array = monthly_clim.values  # (12, nlat, nlon)
features = clim_array.transpose(1, 2, 0).reshape(-1, 12)  # (nlat*nlon, 12)
lats_grid = np.repeat(lat_vals, nlon)
lons_grid = np.tile(lon_vals, nlat)

# Filter grid valid
valid_mask = ~np.isnan(features).all(axis=1)
features = features[valid_mask]
lats_grid = lats_grid[valid_mask]
lons_grid = lons_grid[valid_mask]
print("Valid grid points:", features.shape[0])

# Isi NaN dengan rata-rata kolom
col_mean = np.nanmean(features, axis=0)
inds = np.where(np.isnan(features))
features[inds] = np.take(col_mean, inds[1])

# Normalisasi
scaler = StandardScaler()
X = scaler.fit_transform(features)

# ============================
# 3. SIMPLE SOM IMPLEMENTATION
# ============================
class SimpleSOM:
    def __init__(self, m, n, dim, sigma=1.0, lr=0.5, seed=0):
        self.m = m; self.n = n; self.dim = dim
        self.sigma = sigma; self.lr = lr
        rng = np.random.RandomState(seed)
        self.weights = rng.randn(m, n, dim)

    def winner(self, x):
        d = np.linalg.norm(self.weights - x.reshape(1, 1, self.dim), axis=2)
        idx = np.unravel_index(np.argmin(d), (self.m, self.n))
        return idx

    def train_random(self, data, num_iter=3000):
        for t in range(num_iter):
            x = data[np.random.randint(0, data.shape[0])]
            d = np.linalg.norm(self.weights - x.reshape(1, 1, self.dim), axis=2)
            wi, wj = np.unravel_index(np.argmin(d), (self.m, self.n))
            for i in range(self.m):
                for j in range(self.n):
                    grid_dist = np.sqrt((i - wi) ** 2 + (j - wj) ** 2)
                    h = np.exp(-(grid_dist ** 2) / (2 * (self.sigma ** 2)))
                    self.weights[i, j, :] += self.lr * h * (x - self.weights[i, j, :])
            if (t + 1) % 500 == 0:
                self.lr *= 0.9
                self.sigma *= 0.9

# Latih SOM
som_x, som_y = 6, 6
som = SimpleSOM(som_x, som_y, X.shape[1], sigma=1.0, lr=0.5, seed=42)
som.train_random(X, num_iter=3000)

# Tentukan cluster (unit SOM)
winners = np.array([som.winner(x) for x in X])
cluster_ids = winners[:, 0] * som_y + winners[:, 1]

# ============================
# 4. SMOOTHING SPASIAL
# ============================
coords_xy = np.column_stack([
    np.radians(lons_grid) * 6371000 * np.cos(np.radians(lats_grid)),
    np.radians(lats_grid) * 6371000
])
tree = cKDTree(coords_xy)

def smooth_majority(clust, coords, tree, r_km=80, k=30):
    r = r_km * 1000
    smoothed = clust.copy()
    n = coords.shape[0]
    for i in range(n):
        idx = tree.query_ball_point(coords[i], r)
        if len(idx) < 5:
            _, idx = tree.query(coords[i], k=min(k, n))
            if np.isscalar(idx): idx = [idx]
        vals, counts = np.unique(clust[idx], return_counts=True)
        smoothed[i] = vals[np.argmax(counts)]
    return smoothed

cluster_smooth = smooth_majority(cluster_ids, coords_xy, tree, r_km=80, k=40)

# ============================
# 5. SIMPAN HASIL KE CSV
# ============================
df = pd.DataFrame({'lat': lats_grid, 'lon': lons_grid, 'cluster_raw': cluster_ids, 'cluster_smooth': cluster_smooth})
df_months = pd.DataFrame(features, columns=[f'Bulan_{i+1}' for i in range(12)])
df = pd.concat([df.reset_index(drop=True), df_months.reset_index(drop=True)], axis=1)
df.to_csv("som_clusters_grid.csv", index=False)
print("Saved: som_clusters_grid.csv")

# Statistik cluster
stats = df.groupby('cluster_smooth').agg({**{f'Bulan_{i+1}': 'mean' for i in range(12)}, 'lat': 'count'}) \
    .rename(columns={'lat': 'n_points'}).reset_index().sort_values('n_points', ascending=False)
stats.to_csv("cluster_stats.csv", index=False)
print("Saved: cluster_stats.csv")

# ============================
# 6. VISUALISASI INTERAKTIF
# ============================
# Toggle Raw vs Smoothed
fig = go.Figure()
fig.add_trace(go.Scattermapbox(lat=df['lat'], lon=df['lon'], mode='markers',
                               marker=dict(size=4, color=df['cluster_raw'], colorscale='Viridis', showscale=False),
                               name='Raw', text=[f"Raw {r}, Smooth {s}" for r, s in zip(df['cluster_raw'], df['cluster_smooth'])]))
fig.add_trace(go.Scattermapbox(lat=df['lat'], lon=df['lon'], mode='markers',
                               marker=dict(size=4, color=df['cluster_smooth'], colorscale='Plasma', showscale=False),
                               name='Smoothed', text=[f"Raw {r}, Smooth {s}" for r, s in zip(df['cluster_raw'], df['cluster_smooth'])],
                               visible=False))
fig.update_layout(mapbox=dict(style='carto-positron', center=dict(lat=-2, lon=120), zoom=4),
                  updatemenus=[dict(buttons=[dict(label='Raw', method='update', args=[{'visible': [True, False]}]),
                                             dict(label='Smoothed', method='update', args=[{'visible': [False, True]}])],
                                    direction='left', x=0.1, y=1.05)],
                  margin=dict(l=0, r=0, b=0, t=40))
fig.show()

# Zonasi cluster
fig2 = px.scatter_mapbox(df, lat='lat', lon='lon', color='cluster_smooth', hover_data=['lat', 'lon', 'cluster_smooth'],
                         color_continuous_scale=px.colors.qualitative.Dark24, zoom=4, height=650,
                         title='Zonasi Dominan Cluster (Grid 0.25°)')
fig2.update_layout(mapbox_style='carto-positron', margin=dict(l=0, r=0, b=0, t=40), mapbox_center=dict(lat=-2, lon=120))
fig2.show()

# ============================
# 7. VISUALISASI U-MATRIX
# ============================
weights = som.weights
m, n, d = weights.shape
U = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        w = weights[i, j]
        neigh = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ii, jj = i + di, j + dj
            if 0 <= ii < m and 0 <= jj < n:
                neigh.append(weights[ii, jj])
        if neigh:
            U[i, j] = np.mean([np.linalg.norm(w - nb) for nb in neigh])
plt.figure(figsize=(5, 4))
plt.imshow(U.T, origin='lower')
plt.title("SOM U-Matrix")
plt.colorbar(label='Distance')
plt.tight_layout()
plt.show()

# ============================
# 8. PROFIL BULANAN CLUSTER
# ============================
top_clusters = stats['cluster_smooth'].head(6).tolist()
months = np.arange(1, 13)
for cid in top_clusters:
    row = stats[stats['cluster_smooth'] == cid].iloc[0]
    vals = [row[f'Bulan_{m}'] for m in months]
    plt.figure(figsize=(6, 3))
    plt.plot(months, vals, marker='o')
    plt.xticks(months)
    plt.xlabel('Bulan'); plt.ylabel('Curah Hujan (mm)')
    plt.title(f'Profil Bulanan Cluster {cid} (n={int(row["n_points"])})')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print("Selesai. File keluaran: som_clusters_grid.csv, cluster_stats.csv")
