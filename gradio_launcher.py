import os
import sys
import importlib.util

# Name of the environment variable
ENV_VAR = "GRADIO_APP_PATH"

def main():
    app_path = os.getenv(ENV_VAR)

    if not app_path:
        print(f"❌ Error: Environment variable '{ENV_VAR}' is not set.")
        sys.exit(1)

    if not os.path.isfile(app_path):
        print(f"❌ Error: File at '{app_path}' does not exist.")
        sys.exit(1)

    # Dynamically import the module
    spec = importlib.util.spec_from_file_location("gradio_app", app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Launch the Gradio interface
    if hasattr(module, "launch"):
        module.launch()
    else:
        print("❌ Error: The specified file does not have a 'launch()' function.")
        sys.exit(1)

if __name__ == "__main__":
    main()
