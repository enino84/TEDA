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

### 1. Define a New Analysis Class

Create a new Python file (e.g., `my_analysis.py`) and define a class that inherits from `pyteda.analysis.analysis_base.Analysis`:

```python
# my_analysis.py
import numpy as np
from pyteda.analysis.analysis_base import Analysis

class MyCustomAnalysis(Analysis):
    def __init__(self, param=1):
        self.param = param
        self.Xa = None

    def perform_assimilation(self, background, observation):
        # Custom assimilation logic
        self.Xa = background.ensemble  # For demonstration, no change

    def get_analysis_state(self):
        return self.Xa.mean(axis=1)

    def get_ensemble(self):
        return self.Xa

    def get_error_covariance(self):
        return np.cov(self.Xa)

    def inflate_ensemble(self, inflation_factor):
        mean = self.Xa.mean(axis=1, keepdims=True)
        self.Xa = mean + inflation_factor * (self.Xa - mean)
```

---

### 2. Save It Locally

Save your file in any location accessible from your script or notebook.

---

### 3. Register the Method Dynamically

Use the TEDA registry to make your method available in the `AnalysisFactory`:

```python
from pyteda.analysis.registry import register_analysis_from_file
register_analysis_from_file('my_analysis.py', 'my-analysis')
```

You can now use it as any other method:

```python
from pyteda.analysis.analysis_factory import AnalysisFactory

analysis = AnalysisFactory('my-analysis', param=42).create_analysis()
```


