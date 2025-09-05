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

