import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="SOM Rainfall Analysis", layout="wide")
st.title("🌧️ SOM Clustering untuk Data Curah Hujan (NetCDF)")

# -------------------------
# Utility: pilih nama koordinat yang ada
# -------------------------
def pick_coord_names(ds):
    # kemungkinan nama: 'lat'/'lon' atau 'latitude'/'longitude'
    lat_names = ['lat', 'latitude', 'y']
    lon_names = ['lon', 'longitude', 'x']
    lat_name = next((n for n in lat_names if n in ds.coords), None)
    lon_name = next((n for n in lon_names if n in ds.coords), None)
    return lat_name, lon_name

# -------------------------
# Cache: baca & hitung climatology (agar tidak ulang-ulang saat re-render)
# -------------------------
@st.cache_data(show_spinner=False)
def load_and_climatology(nc_file, var_name):
    ds_local = xr.open_dataset(nc_file)
    da = ds_local[var_name]
    if 'time' not in da.dims:
        raise ValueError("Variabel tidak memiliki dimensi 'time'.")
    clim = da.groupby("time.month").mean("time")  # (month, lat, lon)
    return ds_local, clim

# -------------------------
# Upload file
# -------------------------
uploaded_file = st.file_uploader("Upload file NetCDF (.nc)", type=["nc"])
if uploaded_file is None:
    st.info("Upload file .nc terlebih dahulu (contoh: monthly climatology file).")
    st.stop()

# setelah upload -> pilih var dan lanjut
try:
    ds_tmp = xr.open_dataset(uploaded_file)  # quick open to list variables & coords
except Exception as e:
    st.error(f"Gagal membuka file NetCDF: {e}")
    st.stop()

lat_name, lon_name = pick_coord_names(ds_tmp)
if lat_name is None or lon_name is None:
    st.error("Tidak menemukan koordinat latitude/longitude pada file. Cek nama koordinat (lat/lon).")
    st.stop()

var_name = st.selectbox("Pilih variabel untuk dianalisis:", list(ds_tmp.data_vars.keys()))
st.write(f"Koordinat yg terdeteksi: lat='{lat_name}', lon='{lon_name}'")

# load + climatology (cached)
try:
    ds, clim = load_and_climatology(uploaded_file, var_name)
except Exception as e:
    st.error(f"Error saat menghitung climatology: {e}")
    st.stop()

st.markdown(f"**Climatology** shape: {clim.shape}  (month, lat, lon)")

# ambil array & coords (robust terhadap nama koordinat)
lats = clim[lat_name].values
lons = clim[lon_name].values
data = clim.values  # shape (12, nlat, nlon) assuming month dim first

# -------------------------
# Flatten grid jadi list titik valid
# -------------------------
grid_points = []
features = []
for i_lat, lat in enumerate(lats):
    for j_lon, lon in enumerate(lons):
        vals = data[:, i_lat, j_lon]
        if not np.all(np.isnan(vals)):
            grid_points.append((float(lat), float(lon)))
            features.append(vals)

features = np.array(features)  # (npoints, 12)
grid_points = np.array(grid_points)  # (npoints, 2)
st.write(f"Jumlah grid valid (non-NaN): {features.shape[0]}")

if features.shape[0] == 0:
    st.error("Tidak ada grid valid (semua NaN). Periksa file / variabel.")
    st.stop()

# isi NaN bulanan dengan mean kolom
col_mean = np.nanmean(features, axis=0)
inds = np.where(np.isnan(features))
if inds[0].size > 0:
    features[inds] = np.take(col_mean, inds[1])

# normalisasi
scaler = StandardScaler()
X = scaler.fit_transform(features)

# -------------------------
# UI: parameter SOM
# -------------------------
st.subheader("🧠 Konfigurasi SOM")
som_size = st.slider("Ukuran SOM (N × N)", min_value=4, max_value=10, value=6)
n_iter = st.slider("Jumlah Iterasi SOM", min_value=500, max_value=10000, value=3000, step=500)

# Training button
if st.button("Latih SOM & Tampilkan Hasil"):
    with st.spinner("Melatih SOM — tunggu beberapa saat..."):
        # inisialisasi bobot
        weights = np.random.RandomState(42).rand(som_size, som_size, X.shape[1])
        alpha = 0.5
        sigma0 = max(som_size, som_size) / 2.0

        for t in range(n_iter):
            xi = X[np.random.randint(0, X.shape[0])]
            # cari BMU
            dists = np.linalg.norm(weights - xi, axis=2)
            wi, wj = np.unravel_index(np.argmin(dists), dists.shape)
            # update bobot seluruh grid som
            for i in range(som_size):
                for j in range(som_size):
                    grid_dist = np.sqrt((i - wi) ** 2 + (j - wj) ** 2)
                    h = np.exp(-(grid_dist ** 2) / (2 * (sigma0 ** 2)))
                    weights[i, j] += alpha * h * (xi - weights[i, j])
            # decay
            alpha = 0.5 * (1 - t / n_iter)
            sigma0 = max(som_size, som_size) / 2.0 * (1 - t / n_iter)

    st.success("✅ Training SOM selesai")

    # assign cluster per titik
    cluster_raw = []
    for x in X:
        dists = np.linalg.norm(weights - x, axis=2)
        bi, bj = np.unravel_index(np.argmin(dists), dists.shape)
        cluster_raw.append(int(bi * som_size + bj))
    cluster_raw = np.array(cluster_raw)

    # spatial smoothing (majority vote) menggunakan cKDTree di koordinat lon/lat (derajat)
    tree = cKDTree(grid_points)
    cluster_smooth = np.empty_like(cluster_raw)
    for idx, pt in enumerate(grid_points):
        # radius dalam derajat (approx); gunakan 0.8° default seperti sebelumnya
        nb_idx = tree.query_ball_point(pt, r=0.8)
        if len(nb_idx) == 0:
            cluster_smooth[idx] = cluster_raw[idx]
            continue
        vals, counts = np.unique(cluster_raw[nb_idx], return_counts=True)
        cluster_smooth[idx] = int(vals[np.argmax(counts)])

    # gabungkan menjadi DataFrame untuk plotting + download
    df = pd.DataFrame({
        "lat": grid_points[:, 0],
        "lon": grid_points[:, 1],
        "cluster_raw": cluster_raw,
        "cluster_smooth": cluster_smooth
    })
    # tambahkan bulan asli (mean mm) ke df untuk referensi (opsional)
    for m in range(12):
        df[f"Bulan_{m+1}"] = features[:, m]

    st.subheader("🗺️ Peta Interaktif (Raw vs Smoothed)")
    toggle = st.radio("Tampilkan:", ["Raw", "Smoothed"], horizontal=True)
    col_name = "cluster_raw" if toggle == "Raw" else "cluster_smooth"

    # scatter_mapbox dengan palette kategorikal
    fig = px.scatter_mapbox(df, lat="lat", lon="lon", color=col_name,
                            color_continuous_scale=px.colors.sequential.Viridis,
                            hover_data=["lat", "lon", "cluster_raw", "cluster_smooth"],
                            zoom=4, height=650)
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("💾 Download hasil CSV", df.to_csv(index=False).encode("utf-8"), "som_clusters.csv", "text/csv")

    # Zonasi grid (simple): show same df but with qualitative palette
    st.subheader("🗺️ Zonasi Cluster (Grid 0.25°)")
    fig2 = px.scatter_mapbox(df, lat="lat", lon="lon", color="cluster_smooth",
                             color_continuous_scale=px.colors.qualitative.Dark24,
                             hover_data=["lat", "lon", "cluster_smooth"],
                             zoom=4, height=600)
    fig2.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    # U-matrix (visualize average distance to 4-neighbors)
    st.subheader("U-matrix (SOM distance between neighboring units)")
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
    fig_u, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(U.T, origin="lower")
    ax.set_title("U-matrix")
    st.pyplot(fig_u)

    # Profil bulanan untuk top clusters
    st.subheader("📈 Profil Bulanan: 6 Cluster Terbesar")
    stats = df.groupby("cluster_smooth")[[f"Bulan_{m+1}" for m in range(12)]].mean()
    counts = df["cluster_smooth"].value_counts().sort_values(ascending=False)
    top6 = counts.head(6).index.tolist()
    for cid in top6:
        vals = stats.loc[cid].values
        figp, axp = plt.subplots(figsize=(6, 2.5))
        axp.plot(np.arange(1, 13), vals, marker='o')
        axp.set_xticks(np.arange(1, 13))
        axp.set_xlabel("Bulan")
        axp.set_ylabel("Curah Hujan (unit data)")
        axp.set_title(f"Cluster {cid} (n={counts[cid]})")
        st.pyplot(figp)

    st.success("Analisis selesai. Unduh CSV untuk rincian per-grid.")

