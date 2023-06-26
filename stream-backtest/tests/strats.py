import os
import shutil
import inspect
import importlib

# Create the "strategy" folder if it doesn't exist
folder_name = "/media/eren/PARADIS/SnowCode/Python/MY-REPOS/SnowFreq/stream-backtest/snow_data/strategies"
# os.makedirs(folder_name, exist_ok=True)

# Move all Python files with class definitions to the "strategy" folder
# class_files = [file for file in os.listdir() if file.endswith(".py") and file != "main.py"]
# for file in class_files:
#     shutil.move(file, os.path.join(folder_name, file))

# Import all modules in the "strategy" folder
modules = []
for file in os.listdir(folder_name):
    module_name = os.path.splitext(file)[0]
    module = importlib.import_module(f"snow_data.strategies.{module_name}")
    modules.append(module)

# Retrieve the class names from the imported modules
for module in modules:
    class_names = [name for name, obj in inspect.getmembers(module) if inspect.isclass(obj)]
    print(f"Class names in {module.__name__}: {class_names}")
