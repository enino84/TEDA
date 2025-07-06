# 🧪 Método oficial: 'letkf' con r=2 (usando AnalysisFactory)
import numpy as np
import matplotlib.pyplot as plt

from pyteda.simulation import Simulation
from pyteda.models import Lorenz96
from pyteda.background import Background
from pyteda.observation import Observation
from pyteda.analysis.analysis_factory import AnalysisFactory

model = Lorenz96()
background = Background(model, ensemble_size=20)
observation = Observation(m=32, std_obs=0.01)

params = {'obs_freq': 0.1, 'obs_times': 10, 'inf_fact': 1.04}

# ✅ Crear análisis con parámetro r=2 para LETKF
analysis = AnalysisFactory("letkf", model=model, r=2).create_analysis()

sim = Simulation(model, background, analysis, observation, params=params)
sim.run()

errb, erra = sim.get_errors()
plt.figure(figsize=(10, 6))
plt.plot(np.log10(errb), '-ob', label='Background Error')
plt.plot(np.log10(erra), '-or', label='Analysis Error')
plt.title("LETKF (r=2) – Log10 Errors")
plt.legend()
plt.grid(True)
plt.show()