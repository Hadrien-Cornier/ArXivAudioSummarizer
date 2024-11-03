import sys
from utils.utils import resolve_config
import importlib
import pkgutil
import scripts


def load_pipeline_steps():
    step_functions = {}
    for loader, module_name, is_pkg in pkgutil.iter_modules(scripts.__path__):
        try:
            module = importlib.import_module(f"scripts.{module_name}")
            if hasattr(module, "run"):
                step_functions[module_name] = module.run
                print(f"Loaded module: {module_name}")
            else:
                print(
                    f"Warning: Module '{module_name}' does not have a 'run' function."
                )
        except Exception as e:
            print(f"Error loading module '{module_name}': {e}")
    return step_functions


def main():
    config = resolve_config()
    pipeline_steps = [
        step.strip() for step in config.get("pipeline", "steps").split(",")
    ]

    step_functions = load_pipeline_steps()
    print("Pipeline steps Loaded:", pipeline_steps)

    for step in pipeline_steps:
        print(f"Executing step: {step}")
        if step in step_functions:
            step_functions[step](config=config)
        else:
            print(f"Warning: Unknown pipeline step '{step}'")

    print("Pipeline execution completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}")
        sys.exit(1)
