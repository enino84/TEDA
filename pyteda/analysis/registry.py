ANALYSIS_REGISTRY = {}

def register_analysis(name):
    def decorator(cls):
        ANALYSIS_REGISTRY[name] = cls
        return cls
    return decorator
