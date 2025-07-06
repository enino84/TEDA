# Welcome to TEDA 📚

TEDA is a Python package for teaching and experimenting with **ensemble-based data assimilation** (DA) methods.

It includes toy models (e.g., Lorenz 96), prebuilt DA methods, and tools to compare their performance visually and numerically.

# API Reference – Analysis Methods

TEDA includes a growing set of ensemble-based data assimilation methods, all of which inherit from the base `Analysis` interface. These methods can be instantiated directly or dynamically via the `AnalysisFactory`.

Below is a summary of available methods and the base abstract class:

---

## Included Methods

| Class                          | Description                                      | Reference                                                  |
| ------------------------------ | ------------------------------------------------ | ---------------------------------------------------------- |
| AnalysisEnKFShrinkagePrecision | Compute the precision matrix for EnKF Shrinkage Precision. | Nino-Ruiz, Elias D., and Adrian Sandu. "Ensemble Kalman filter implementations based on shrinkage covariance matrix estimation." Ocean Dynamics 65 (2015): 1423-1439.                                                        |
| AnalysisEnKFModifiedCholesky   | Compute the precision matrix for EnKF Modified Cholesky. | Nino-Ruiz, Elias D., Adrian Sandu, and Xinwei Deng. "An ensemble Kalman filter implementation based on modified Cholesky decomposition for inverse covariance matrix estimation." SIAM Journal on Scientific Computing 40.2 (2018): A867-A886.                                                       |
| AnalysisEnKFNaive              | Perform assimilation using Naive EnKF method.    | Nino Ruiz, Elias D., Adrian Sandu, and Jeffrey Anderson. "An efficient implementation of the ensemble Kalman filter based on an iterative Sherman–Morrison formula." Statistics and Computing 25 (2015): 561-577.                                                       |
| AnalysisEnSRF                  | Perform assimilation using EnSRF method.          | Tippett, Michael K., et al. "Ensemble square root filters." Monthly weather review 131.7 (2003): 1485-1490.                                                     |
| AnalysisETKF                   | Perform assimilation using ETKF method.           | Bishop, Craig H., Brian J. Etherton, and Sharanya J. Majumdar. "Adaptive sampling with the ensemble transform Kalman filter. Part I: Theoretical aspects." Monthly weather review 129.3 (2001): 420-436.                                                        |
| AnalysisLETKF                  | Perform assimilation using LETKF method.          | Hunt, Brian R., Eric J. Kostelich, and Istvan Szunyogh. "Efficient data assimilation for spatiotemporal chaos: A local ensemble transform Kalman filter." Physica D: Nonlinear Phenomena 230.1-2 (2007): 112-126.                                                      |
| AnalysisLEnKF                   | Perform assimilation using LEnKF method.          |Ott, Edward, et al. "A local ensemble Kalman filter for atmospheric data assimilation." Tellus A: Dynamic Meteorology and Oceanography 56.5 (2004): 415-428.                                                    |
| AnalysisEnKFBLoc               | Perform assimilation using EnKF B-Loc method.     | Greybush, Steven J., et al. "Balance and ensemble Kalman filter localization techniques." Monthly Weather Review 139.2 (2011): 511-522.                                                      |
| AnalysisEnKFCholesky           | Perform assimilation using EnKF Cholesky decomposition. | Mandel, Jan. Efficient implementation of the ensemble Kalman filter. University of Colorado at Denver and Health Sciences Center, Center for Computational Mathematics, 2006.                                                        |
| AnalysisEnKF                    | Perform assimilation using EnKF full covariance matrix. | Evensen, Geir. Data assimilation: the ensemble Kalman filter. Vol. 2. Berlin: springer, 2009.                                                        |
| Analysis                        | Perform assimilation given background and observations. | Abstract class to define general methods for all assimilation steps.                                                          |

**Analysis (Abstract class for analysis methods)**

| Method                 | Description                                      |
| ---------------------- | ------------------------------------------------ |
| perform_assimilation   | Perform the assimilation step given the background and observations. It takes the following parameters: <br>- **background**: Background object representing the background ensemble or state.<br>- **observation**: Observation object representing the observation used for assimilation. |
| get_analysis_state     | Return the computed column mean of the ensemble Xa. It doesn't take any parameters. |
| get_ensemble           | Return the ensemble Xa. It doesn't take any parameters. |
| get_error_covariance   | Return the computed covariance matrix of the ensemble Xa. It doesn't take any parameters. |
| inflate_ensemble       | Compute the new ensemble Xa given the inflation factor. It takes the following parameter: <br>- **inflation_factor**: Double number indicating the inflation factor to be applied to the ensemble. |

**AnalysisEnKF (Analysis EnKF full covariance matrix)**

| Method                 | Description                                      |
| ---------------------- | ------------------------------------------------ |
| perform_assimilation   | Perform the assimilation step given the background and observations. It takes the following parameters: <br>- **background**: Background object representing the background ensemble or state.<br>- **observation**: Observation object representing the observation used for assimilation. |
| get_analysis_state     | Return the computed column mean of the ensemble Xa. It doesn't take any parameters. |
| get_ensemble           | Return the ensemble Xa. It doesn't take any parameters. |
| get_error_covariance   | Return the computed covariance matrix of the ensemble Xa. It doesn't take any parameters. |
| inflate_ensemble       | Compute the new ensemble Xa given the inflation factor. It takes the following parameter: <br>- **inflation_factor**: Double number indicating the inflation factor to be applied to the ensemble. |

**AnalysisEnKFBLoc (Analysis EnKF B-Loc)**

| Method                 | Description                                      |
| ---------------------- | ------------------------------------------------ |
| perform_assimilation   | Perform the assimilation step given the background and observations. It takes the following parameters: <br>- **background**: Background object representing the background ensemble or state.<br>- **observation**: Observation object representing the observation used for assimilation. |
| get_analysis_state     | Return the computed column mean of the ensemble Xa. It doesn't take any parameters. |
| get_ensemble           | Return the ensemble Xa. It doesn't take any parameters. |
| get_error_covariance   | Return the computed covariance matrix of the ensemble Xa. It doesn't take any parameters. |
| inflate_ensemble       | Compute the new ensemble Xa given the inflation factor. It takes the following parameter: <br>- **inflation_factor**: Double number indicating the inflation factor to be applied to the ensemble. |

**AnalysisEnKFCholesky (EnKF implementation Cholesky)**

| Method                 | Description                                      |
| ---------------------- | ------------------------------------------------ |
| perform_assimilation   | Perform the assimilation step given the background and observations. It takes the following parameters: <br>- **background**: Background object representing the background ensemble or state.<br>- **observation**: Observation object representing the observation used for assimilation. |
| get_analysis_state     | Return the computed column mean of the ensemble Xa. It doesn't take any parameters. |
| get_ensemble           | Return the ensemble Xa. It doesn't take any parameters. |
| get_error_covariance   | Return the computed covariance matrix of the ensemble Xa. It doesn't take any parameters. |
| inflate_ensemble       | Compute the new ensemble Xa given the inflation factor. It takes the following parameter: <br>- **inflation_factor**: Double number indicating the inflation factor to be applied to the ensemble. |

**AnalysisEnKFModifiedCholesky (Analysis EnKF Modified Cholesky decomposition)**

| Method                 | Description                                      |
| ---------------------- | ------------------------------------------------ |
| get_precision_matrix   | Returns the computed precision matrix. It takes the following parameters: <br>- **DX**: Deviation matrix.<br>- **regularization_factor**: Value used as alpha in the ridge model. |
| perform_assimilation   | Perform the assimilation step given the background and observations. It takes the following parameters: <br>- **background**: Background object representing the background ensemble or state.<br>- **observation**: Observation object representing the observation used for assimilation. |
| get_analysis_state     | Return the computed column mean of the ensemble Xa. It doesn't take any parameters. |
| get_ensemble           | Return the ensemble Xa. It doesn't take any parameters. |
| get_error_covariance   | Return the computed covariance matrix of the ensemble Xa. It doesn't take any parameters. |
| inflate_ensemble       | Compute the new ensemble Xa given the inflation factor. It takes the following parameter: <br>- **inflation_factor**: Double number indicating the inflation factor to be applied to the ensemble. |

**AnalysisEnKFNaive (Analysis EnKF Naive)**

| Method                 | Description                                      |
| ---------------------- | ------------------------------------------------ |
| perform_assimilation   | Perform the assimilation step given the background and observations. It takes the following parameters: <br>- **background**: Background object representing the background ensemble or state.<br>- **observation**: Observation object representing the observation used for assimilation. |
| get_analysis_state     | Return the computed column mean of the ensemble Xa. It doesn't take any parameters. |
| inflate_ensemble       | Compute the new ensemble Xa given the inflation factor. It takes the following parameter: <br>- **inflation_factor**: Double number indicating the inflation factor to be applied to the ensemble. |
| get_ensemble           | Return the ensemble Xa. It doesn't take any parameters. |
| get_error_covariance   | Return the computed covariance matrix of the ensemble Xa. It doesn't take any parameters. |

**AnalysisEnKFShrinkagePrecision (Analysis EnKF Shrinkage Precision)**

| Method                 | Description                                      |
| ---------------------- | ------------------------------------------------ |
| get_precision_matrix   | Returns the computed precision matrix. It takes the following parameters: <br>- **DX**: Deviation matrix.<br>- **regularization_factor**: Value used as alpha in the ridge model. |
| perform_assimilation   | Perform the assimilation step given the background and observations. It takes the following parameters: <br>- **background**: Background object representing the background ensemble or state.<br>- **observation**: Observation object representing the observation used for assimilation. |
| get_analysis_state     | Return the computed column mean of the ensemble Xa. It doesn't take any parameters. |
| get_ensemble           | Return the ensemble Xa. It doesn't take any parameters. |
| get_error_covariance   | Return the computed covariance matrix of the ensemble Xa. It doesn't take any parameters. |
| inflate_ensemble       | Compute the new ensemble Xa given the inflation factor. It takes the following parameter: <br>- **inflation_factor**: Double number indicating the inflation factor to be applied to the ensemble. |

**AnalysisEnSRF (Analysis EnSRF)**

| Method                 | Description                                      |
| ---------------------- | ------------------------------------------------ |
| perform_assimilation   | Perform the assimilation step given the background and observations. It takes the following parameters: <br>- **background**: Background object representing the background ensemble or state.<br>- **observation**: Observation object representing the observation used for assimilation. |
| get_analysis_state     | Return the computed column mean of the ensemble Xa. It doesn't take any parameters. |
| get_ensemble           | Return the ensemble Xa. It doesn't take any parameters. |
| get_error_covariance   | Return the computed covariance matrix of the ensemble Xa. It doesn't take any parameters. |
| inflate_ensemble       | Compute the new ensemble Xa given the inflation factor. It takes the following parameter: <br>- **inflation_factor**: Double number indicating the inflation factor to be applied to the ensemble. |

**AnalysisETKF (Analysis Ensemble Transform Kalman Filter (ETKF))**

| Method                 | Description                                      |
| ---------------------- | ------------------------------------------------ |
| perform_assimilation   | Perform the assimilation step given the background and observations. It takes the following parameters: <br>- **background**: Background object representing the background ensemble or state.<br>- **observation**: Observation object representing the observation used for assimilation. |
| get_analysis_state     | Return the computed column mean of the ensemble Xa. It doesn't take any parameters. |
| get_ensemble           | Return the ensemble Xa. It doesn't take any parameters. |
| get_error_covariance   | Return the computed covariance matrix of the ensemble Xa. It doesn't take any parameters. |
| inflate_ensemble       | Compute the new ensemble Xa given the inflation factor. It takes the following parameter: <br>- **inflation_factor**: Double number indicating the inflation factor to be applied to the ensemble. |

**AnalysisLEnKF (Analysis LEnKF)**

| Method                 | Description                                      |
| ---------------------- | ------------------------------------------------ |
| perform_assimilation   | Perform the assimilation step given the background and observations. It takes the following parameters: <br>- **background**: Background object representing the background ensemble or state.<br>- **observation**: Observation object representing the observation used for assimilation. |
| get_analysis_state     | Return the computed column mean of the ensemble Xa. It doesn't take any parameters. |
| inflate_ensemble       | Compute the new ensemble Xa given the inflation factor. It takes the following parameter: <br>- **inflation_factor**: Double number indicating the inflation factor to be applied to the ensemble. |
| get_ensemble           | Return the ensemble Xa. It doesn't take any parameters. |
| get_error_covariance   | Return the computed covariance matrix of the ensemble Xa. It doesn't take any parameters. |

**AnalysisLETKF (Analysis Local Ensemble Transform Kalman Filter (LETKF))**

| Method                 | Description                                      |
| ---------------------- | ------------------------------------------------ |
| perform_assimilation   | Perform the assimilation step given the background and observations. It takes the following parameters: <br>- **background**: Background object representing the background ensemble or state.<br>- **observation**: Observation object representing the observation used for assimilation. |
| get_analysis_state     | Return the computed column mean of the ensemble Xa. It doesn't take any parameters. |
| inflate_ensemble       | Compute the new ensemble Xa given the inflation factor. It takes the following parameter: <br>- **inflation_factor**: Double number indicating the inflation factor to be applied to the ensemble. |
| get_ensemble           | Return the ensemble Xa. It doesn't take any parameters. |
| get_error_covariance   | Return the computed covariance matrix of the ensemble Xa. It doesn't take any parameters. |



---

## Interface: `Analysis`

All methods must implement the following interface:

```python
class Analysis(ABC):
    @abstractmethod
    def perform_assimilation(self, background, observation):
        pass

    @abstractmethod
    def get_analysis_state(self):
        pass

    @abstractmethod
    def get_ensemble(self):
        pass

    @abstractmethod
    def get_error_covariance(self):
        pass

    @abstractmethod
    def inflate_ensemble(self, inflation_factor):
        pass
```

---

## ✨ Creating Your Own Analysis Method

TEDA is designed to be easily extensible. You can add your own data assimilation method by implementing a custom class that inherits from the abstract `Analysis` class.


### ✅ Step 1: Define and register your custom method

Save this file as `user_methods/my_enkf.py`:

```python
# user_methods/my_enkf.py

from pyteda.analysis.analysis_core import Analysis
from pyteda.analysis.registry import register_analysis

@register_analysis("my_enkf")  # ✅ Decorator to register the method
class MyEnKF(Analysis):

    def __init__(self, **kwargs):
      pass

    def perform_assimilation(self, background, observation):
        # Simple assimilation: apply scaled innovation
        HXb = observation.H @ background.Xb
        innovations = observation.y[:, None] - HXb
        self.Xa = background.Xb + 0.1 * (observation.H.T @ innovations)
        return self.Xa

    def get_analysis_state(self):
        return self.Xa.mean(axis=1)

    def get_ensemble(self):
        return self.Xa

    def get_error_covariance(self):
        Xa_mean = self.Xa.mean(axis=1, keepdims=True)
        return ((self.Xa - Xa_mean) @ (self.Xa - Xa_mean).T) / (self.Xa.shape[1] - 1)

    def inflate_ensemble(self, inflation_factor):
        mean = self.Xa.mean(axis=1, keepdims=True)
        self.Xa = mean + inflation_factor * (self.Xa - mean)
```

---

### 📁 Step 2: Use your method via the factory

```python
import numpy as np
import matplotlib.pyplot as plt

from pyteda.simulation import Simulation
from pyteda.models import Lorenz96
from pyteda.background import Background
from pyteda.observation import Observation
from pyteda.analysis.analysis_factory import AnalysisFactory

# 🔄 Import your custom class to trigger the registration
import user_methods.my_enkf

# Setup
model = Lorenz96()
background = Background(model, ensemble_size=20)
observation = Observation(m=32, std_obs=0.01)
params = {'obs_freq': 0.1, 'obs_times': 10, 'inf_fact': 1.04}

# ✅ Use your registered custom method via factory
analysis = AnalysisFactory("my_enkf", model=model).create_analysis()

# Run simulation
sim = Simulation(model, background, analysis, observation, params=params)
sim.run()

# Visualize error
errb, erra = sim.get_errors()
plt.figure(figsize=(10, 6))
plt.plot(np.log10(errb), '-ob', label='Background Error')
plt.plot(np.log10(erra), '-or', label='Analysis Error')
plt.title("Custom Method (MyEnKF) – Log10 Errors")
plt.legend()
plt.grid(True)
plt.show()
```

---

## More examples here [Google Colab](https://colab.research.google.com/drive/1Ts67iE19KtMuWd0CF8_GGGqE3wRQjC2G?usp=sharing)