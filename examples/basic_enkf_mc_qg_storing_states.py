import numpy as np
import matplotlib.pyplot as plt

from pyteda.simulation import Simulation
from pyteda.models import QGModel
from pyteda.background import Background
from pyteda.observation import Observation
from pyteda.analysis.analysis_factory import AnalysisFactory

# -------------------------------
# 1. Model and object configuration
# -------------------------------
model = QGModel(N=32, dt=0.001, F=1600.0)
background = Background(model, ensemble_size=40)
observation = Observation(m=400, std_obs=0.01)

params = {
    'obs_freq': 0.1,
    'obs_times': 20,
    'inf_fact': 1.04,
    'store_back_state': True,
    'store_post_state': True,
    'store_ref_state': True,
    'store_state_at': [0, 10, 19]  # ✅ Store only at these steps
}

analysis = AnalysisFactory("enkf-modified-cholesky", model=model).create_analysis()

# -------------------------------
# 2. Run the simulation
# -------------------------------
sim = Simulation(model, background, analysis, observation, params=params)
sim.run()

# -------------------------------
# 3. Retrieve and reshape stored states
# -------------------------------
saved = sim.get_saved_states()
q_truth = np.array(saved["truth"])         # shape: (3, N*N)
q_back = np.array(saved["background"])     # shape: (3, N*N)
q_analysis = np.array(saved["analysis"])   # shape: (3, N*N)
stored_steps = saved["steps"]              # e.g., [0, 10, 19]

N = model.N
labels = [f"Step {s}" for s in stored_steps]

# -------------------------------
# 4. Plot the 3×3 state matrix
# -------------------------------
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
states = [q_truth, q_back, q_analysis]
titles = ["True", "Background", "Analysis"]

for row, state_group in enumerate(states):
    for col, step in enumerate(stored_steps):
        field = state_group[col].reshape(N, N)
        im = axes[row, col].contourf(field, levels=40, cmap="RdBu_r")
        if row == 0:
            axes[row, col].set_title(labels[col], fontsize=12)
        if col == 0:
            axes[row, col].set_ylabel(titles[row], fontsize=12)
        plt.colorbar(im, ax=axes[row, col], orientation='vertical', shrink=0.8)

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.suptitle("QGModel – Vorticity States at Selected Timesteps", fontsize=16, y=1.02)
plt.show()
