import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from pyteda.simulation import Simulation
from pyteda.models import QGModel
from pyteda.background import Background
from pyteda.observation import Observation
from pyteda.analysis.analysis_factory import AnalysisFactory

# 1. Setup
model = QGModel(N=32, dt=0.001, F=1600.0)
background = Background(model, ensemble_size=20)
observation = Observation(m=400, std_obs=0.01)

params = {
    'obs_freq': 1,
    'end_time': 20,
    'inf_fact': 1.04,
    'store_back_state': True,
    'store_post_state': True,
    'store_ref_state': True,
    'store_state_at': [0, 10, 20]
}

analysis = AnalysisFactory("enkf-modified-cholesky", model=model).create_analysis()
sim = Simulation(model, background, analysis, observation, params=params)
sim.run()

# 2. Load states
saved = sim.get_saved_states()
q_full_truth = np.array(saved["truth"])
q_full_back = np.array(saved["background"])
q_full_analysis = np.array(saved["analysis"])
stored_steps = saved["steps"]

N = model.N
lat = np.linspace(-90, 90, N)
lon = np.linspace(-180, 180, N)
Lon, Lat = np.meshgrid(lon, lat)

# ✅ Función para dividir estado completo en q y ψ
def split_q_psi(full_state_data):
    q_data = []
    psi_data = []
    for flat in full_state_data:
        q_data.append(flat[:N*N])
        psi_data.append(flat[N*N:])
    return np.array(q_data), np.array(psi_data)

# ✅ Separar para cada conjunto
q_truth, psi_truth = split_q_psi(q_full_truth)
q_back, psi_back = split_q_psi(q_full_back)
q_analysis, psi_analysis = split_q_psi(q_full_analysis)

states_q = [q_truth, q_back, q_analysis]
states_psi = [psi_truth, psi_back, psi_analysis]
titles = ["True", "Background", "Analysis"]
labels = [f"Step {s}" for s in stored_steps]

# ✅ Función para graficar conjuntos (q o psi)
def plot_field(states, field_name):
    fig, axes = plt.subplots(3, 3, figsize=(20, 10),
                             subplot_kw={'projection': ccrs.Robinson()})
    for row, state_group in enumerate(states):
        for col, step in enumerate(stored_steps):
            ax = axes[row, col]
            ax.set_global()
            ax.coastlines()
            ax.gridlines(draw_labels=False, linewidth=0.5, linestyle='--', color='gray')
            field = state_group[col].reshape(N, N)
            im = ax.contourf(Lon, Lat, field, 40, transform=ccrs.PlateCarree(), cmap='RdBu_r')
            if row == 0:
                ax.set_title(labels[col], fontsize=12)
            if col == 0:
                ax.text(-0.1, 0.5, titles[row], va='center', ha='right', transform=ax.transAxes, fontsize=12)
            plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)

    plt.suptitle(f"QGModel – {field_name} on a Globe", fontsize=16, y=0.95)
    plt.tight_layout()
    plt.show()

# 3. Plot q
plot_field(states_q, "Vorticity q")

# 4. Plot ψ
plot_field(states_psi, "Streamfunction ψ")
