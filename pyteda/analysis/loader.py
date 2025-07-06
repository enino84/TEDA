import importlib.util
import pathlib
import sys

def load_user_methods(folder_path: str):
    folder = pathlib.Path(folder_path).resolve()
    for py_file in folder.glob("*.py"):
        module_name = py_file.stem
        spec = importlib.util.spec_from_file_location(f"user_methods.{module_name}", py_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
