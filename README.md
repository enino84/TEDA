# 🧠 TEDA – Toolbox for Ensemble Data Assimilation

We are thrilled to present TEDA, a cutting-edge Python toolbox designed to facilitate the teaching and learning of ensemble-based data assimilation (DA). TEDA caters to the needs of educators and learners alike, offering a comprehensive platform to explore the captivating world of meteorological anomalies, climate change, and DA methodologies.

TEDA stands out as an exceptional resource in the landscape of DA software, as it prioritizes the educational experience. While operational software tends to focus on practical applications, TEDA empowers undergraduate and graduate students by providing an intuitive platform to grasp the intricacies of ensemble-based DA concepts and foster enthusiasm for scientific exploration.

Equipped with a diverse range of features and functionalities, TEDA enriches the learning process by offering powerful visualization tools. Students can delve into error statistics, model errors, observational errors, error distributions, and the dynamic evolution of errors. These visualizations provide multiple perspectives on numerical results, enabling students to develop a deep understanding of ensemble-based DA principles.

TEDA offers numerous ensemble-based DA methods, allowing students to explore an extensive range of techniques. From the stochastic ensemble Kalman filter (EnKF) to dual EnKF formulations, EnKF via Cholesky decomposition, EnKF based on modified Cholesky decomposition, EnKF based on B-localization, and beyond, TEDA equips learners with a comprehensive toolkit to study and compare various DA methodologies.

TEDA's foundation is rooted in the Object-Oriented Programming (OOP) paradigm, enabling effortless integration of new techniques and models. This ensures that TEDA remains at the forefront of the rapidly evolving field of ensemble-based DA, allowing educators and researchers to incorporate the latest advancements into their teaching and experimentation.

Embark on a captivating journey of discovery and knowledge with TEDA as it unveils the secrets of data assimilation. Simulate various DA scenarios, experiment with different model configurations, and witness firsthand the transformative impact of ensemble-based DA methods on forecast accuracy and data analysis.

Unlock your true potential in the realm of data assimilation with TEDA. Whether you are an enthusiastic student or a dedicated educator, TEDA stands as your ultimate companion in unraveling the complexities of ensemble-based DA.

Keywords: Data Assimilation, Ensemble Kalman Filter, Education, Python.

### 🧪 Toy Models for Data Assimilation

To enhance the learning experience, **TEDA** is built around **toy models** that exhibit chaotic behavior under specific configurations. These models are ideal for experimenting with and validating various Data Assimilation (DA) techniques in a hands-on, controlled setting.

Currently, TEDA includes the **Lorenz 96 model** (40 variables), a well-known benchmark in DA research.

> 📌 *Support for other classic chaotic systems like the Duffing equation and Lorenz 63 may be added in future versions.*

---

## 📦 Installation

Aquí tienes una versión mejor redactada y estructurada para la sección de instalación y uso rápido:

---

## 🚀 Installation and Quick Start

### ✅ From PyPI

To install the latest stable version:

```bash
pip install pyteda
```

### 🧪 From Source (for development)

Clone the repository and install in editable mode:

```bash
git clone https://github.com/your-username/pyteda.git
cd pyteda
pip install -e .
```

### 🛠️ Recommended Setup: Virtual Environment

We suggest using a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 Example Usage

You can run a complete simulation workflow using:

```bash
python examples/basic_template.py
```

Or explore the file directly: [`examples/basic_template.py`](examples/basic_template.py)

---

### ▶️ Run the basic example

Once installed, you can run the full example with:

```bash
python -m examples.basic_template
```

This will initialize the Lorenz96 model, run an ensemble-based data assimilation simulation, and display the log-relative errors for both the background and analysis states.

---

### Supported Methods

The `pyteda` package supports a wide range of ensemble-based data assimilation methods including EnKF, LETKF, ETKF, and shrinkage-based filters. All methods are accessible via the `AnalysisFactory`.

📄 See [Supported Methods](docs/api_reference.md) for a full list and references.

---

## 🚀 Quickstart

Here's a minimal example using the **LETKF** method with the built-in Lorenz96 model:

```python
from pyteda.simulation import Simulation
from pyteda.models import Lorenz96
from pyteda.background import Background
from pyteda.observation import Observation
from pyteda.analysis.analysis_factory import AnalysisFactory

model = Lorenz96()
background = Background(model, ensemble_size=20)
observation = Observation(m=32, std_obs=0.01)

analysis = AnalysisFactory("letkf", model=model).create_analysis()

params = {'obs_freq': 0.1, 'obs_times': 10, 'inf_fact': 1.04}
sim = Simulation(model, background, analysis, observation, params=params)
sim.run()

errb, erra = sim.get_errors()
```

📈 For visualization and full working examples, check the [`examples/`](examples/) folder.

---

## 🎓 Citation

If you use **TEDA** in your teaching or research, please cite:

**Nino-Ruiz, E.D., Valbuena, S.R. (2022)**
*TEDA: A Computational Toolbox for Teaching Ensemble Based Data Assimilation*.
[📖 ICCS 2022, Springer](https://doi.org/10.1007/978-3-031-08760-8_60)

```bibtex
@inproceedings{nino2022teda,
  title={TEDA: A Computational Toolbox for Teaching Ensemble Based Data Assimilation},
  author={Nino-Ruiz, Elias D. and Valbuena, Sebastian Racedo},
  booktitle={Computational Science – ICCS 2022},
  series={Lecture Notes in Computer Science},
  volume={13353},
  pages={787--801},
  year={2022},
  publisher={Springer, Cham},
  doi={10.1007/978-3-031-08760-8_60}
}
```

---

## 📚 More Resources

* 🧪 Explore [examples/](examples/) to try different filters and models.
* 📖 Full documentation available in the `docs/` folder.
* ➕ Want to add your own method? TEDA supports [custom DA methods via registry](docs/custom_methods.md).

---

[![PyPI version](https://badge.fury.io/py/pyteda.svg)](https://pypi.org/project/pyteda/)
[![Cite TEDA](https://img.shields.io/badge/Cite-TEDA-blue.svg)](https://doi.org/10.1007/978-3-031-08760-8_60)

---

## Developed by

* Elías D. Niño-Ruiz
* [https://enino84.github.io/](https://enino84.github.io/)
* elias.d.nino@gmail.com

