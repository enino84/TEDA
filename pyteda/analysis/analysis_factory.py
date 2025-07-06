# analysis_factory.py

# 🔁 Importa todos los archivos de métodos oficiales para que se auto-registren
from . import (
    analysis_enkf,
    analysis_enkf_bloc,
    analysis_enkf_modified_cholesky,
    analysis_enkf_cholesky,
    analysis_enkf_naive,
    analysis_lenkf,
    analysis_ensrf,
    analysis_etkf,
    analysis_letkf,
    analysis_enkf_shrinkage_precision,
)

from .registry import ANALYSIS_REGISTRY


class AnalysisFactory:
    def __init__(self, method='enkf', **kwargs):
        if method not in ANALYSIS_REGISTRY:
            raise ValueError(f"Invalid method name: '{method}'\n"
                             f"Available: {list(ANALYSIS_REGISTRY.keys())}")
        self.analysis_type = ANALYSIS_REGISTRY[method]
        self.analysis_kwargs = kwargs

    def create_analysis(self):
        return self.analysis_type(**self.analysis_kwargs)
