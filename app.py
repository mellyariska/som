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
file_path = "data_kompres.nc"  # Ganti sesuai lokasi file kamu
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


import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="SOM Rainfall Analysis", layout="wide")

st.title("🌧️ SOM Clustering untuk Data Curah Hujan (NetCDF)")

# ==============================
# 1. Upload File NetCDF
# ==============================
uploaded_file = st.file_uploader("Upload file NetCDF (.nc)", type=["nc"])

if uploaded_file is not None:
    try:
        ds = xr.open_dataset(uploaded_file)
        st.success(f"File berhasil dimuat! Dimensi: {list(ds.dims.keys())}")
        
        # Pilih variabel
        var_name = st.selectbox("Pilih variabel:", list(ds.data_vars.keys()))
        da = ds[var_name]
        
        st.write(f"Dimensi variabel: {list(da.dims)}")
        
        # ==============================
        # 2. Hitung Climatology 12 Bulan
        # ==============================
        st.subheader("📊 Hitung Climatology")
        if "time" in da.dims:
            clim = da.groupby("time.month").mean("time")
            st.write(f"Climatology shape: {clim.shape}")
        else:
            st.error("Data tidak memiliki dimensi waktu 'time'.")
            st.stop()
        
        # Ambil data numpy
        lats = clim["latitude"].values
        lons = clim["longitude"].values
        data = clim.values  # shape: (12, lat, lon)
        
        # Buat grid flatten
        nlat, nlon = len(lats), len(lons)
        grid_points = []
        features = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                vals = data[:, i, j]
                if not np.isnan(vals).all():
                    grid_points.append((lat, lon))
                    features.append(vals)
        
        features = np.array(features)
        grid_points = np.array(grid_points)
        
        st.write(f"Jumlah grid valid: {len(features)}")
        
        # Normalisasi
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        
        # ==============================
        # 3. Training SOM
        # ==============================
        st.subheader("🧠 Training Self-Organizing Map (SOM)")
        som_size = st.slider("Ukuran SOM (NxN)", 4, 10, 6)
        n_iter = st.slider("Jumlah Iterasi", 1000, 5000, 2000)
        
        if st.button("Latih SOM"):
            # Inisialisasi SOM
            som_grid = (som_size, som_size)
            weights = np.random.rand(som_size, som_size, X.shape[1])
            
            # Learning Rate dan Decay
            alpha = 0.5
            sigma = max(som_size, som_size) / 2
            
            # Koordinat SOM
            coords = np.array([(i, j) for i in range(som_size) for j in range(som_size)])
            
            for t in range(n_iter):
                idx = np.random.randint(0, X.shape[0])
                x = X[idx]
                
                # Cari BMU
                dists = np.linalg.norm(weights - x, axis=2)
                bmu_idx = np.unravel_index(np.argmin(dists), dists.shape)
                
                # Update bobot
                for i in range(som_size):
                    for j in range(som_size):
                        dist = np.linalg.norm(np.array([i, j]) - np.array(bmu_idx))
                        h = np.exp(-dist**2 / (2 * sigma**2))
                        weights[i, j] += alpha * h * (x - weights[i, j])
                
                # Decay
                alpha = 0.5 * (1 - t / n_iter)
                sigma = max(som_size, som_size) / 2 * (1 - t / n_iter)
            
            st.success("✅ Training selesai!")
            
            # Assign cluster
            cluster_ids = []
            for x in X:
                dists = np.linalg.norm(weights - x, axis=2)
                bmu_idx = np.unravel_index(np.argmin(dists), dists.shape)
                cluster_ids.append(bmu_idx[0] * som_size + bmu_idx[1])
            
            cluster_ids = np.array(cluster_ids)
            
            # Buat smoothing (majority filter)
            tree = cKDTree(grid_points)
            cluster_smooth = []
            for idx, pt in enumerate(grid_points):
                neighbors = tree.query_ball_point(pt, r=0.8)  # radius 0.8°
                vals = cluster_ids[neighbors]
                unique, counts = np.unique(vals, return_counts=True)
                cluster_smooth.append(unique[np.argmax(counts)])
            
            cluster_smooth = np.array(cluster_smooth)
            
            # ==============================
            # 4. Peta Interaktif
            # ==============================
            st.subheader("🗺️ Peta Interaktif (Raw vs Smoothed)")
            df = pd.DataFrame({
                "lat": grid_points[:,0],
                "lon": grid_points[:,1],
                "cluster_raw": cluster_ids,
                "cluster_smooth": cluster_smooth
            })
            
            toggle = st.radio("Tampilkan:", ["Raw", "Smoothed"])
            col_name = "cluster_raw" if toggle == "Raw" else "cluster_smooth"
            
            fig = px.scatter_mapbox(
                df, lat="lat", lon="lon", color=col_name,
                color_continuous_scale="Viridis",
                zoom=4, height=600
            )
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)
            
            # ==============================
            # 5. Download CSV
            # ==============================
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Download Hasil CSV", csv, "som_clusters.csv", "text/csv")
    
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")


