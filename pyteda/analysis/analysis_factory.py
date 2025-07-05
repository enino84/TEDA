from .analysis_enkf import AnalysisEnKF
from .analysis_enkf_bloc import AnalysisEnKFBLoc
from .analysis_enkf_modified_cholesky import AnalysisEnKFModifiedCholesky
from .analysis_enkf_cholesky import AnalysisEnKFCholesky
from .analysis_enkf_naive import AnalysisEnKFNaive
from .analysis_lenkf import AnalysisLEnKF
from .analysis_ensrf import AnalysisEnSRF
from .analysis_etkf import AnalysisETKF
from .analysis_letkf import AnalysisLETKF
from .analysis_enkf_shrinkage_precision import AnalysisEnKFShrinkagePrecision

class AnalysisFactory:
    
    def __init__(self, method='enkf', **kwargs):
        if method == 'enkf':
            self.analysis_type = AnalysisEnKF
        elif method == 'enkf-b-loc':
            self.analysis_type = AnalysisEnKFBLoc
        elif method == 'enkf-modified-cholesky':
            self.analysis_type = AnalysisEnKFModifiedCholesky
        elif method == 'enkf-cholesky':
            self.analysis_type = AnalysisEnKFCholesky
        elif method == 'enkf-naive':
            self.analysis_type = AnalysisEnKFNaive
        elif method == 'lenkf':
            self.analysis_type = AnalysisLEnKF
        elif method == 'ensrf':
            self.analysis_type = AnalysisEnSRF
        elif method == 'etkf':
            self.analysis_type = AnalysisETKF
        elif method == 'letkf':
            self.analysis_type = AnalysisLETKF
        elif method == 'enkf-shrinkage-precision':
            self.analysis_type = AnalysisEnKFShrinkagePrecision
        else:
            raise ValueError(f"Invalid method name: {method}")

        self.analysis_kwargs = kwargs

    def create_analysis(self):
        return self.analysis_type(**self.analysis_kwargs)
