# recursively import every submodule at runtime
# source: https://stackoverflow.com/questions/3365740/how-to-import-all-submodules
import pkgutil

__all__ = []
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    _module = loader.find_module(module_name).load_module(module_name)
    globals()[module_name] = _module
