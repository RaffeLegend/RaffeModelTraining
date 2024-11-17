import argparse
from src.options.yaml_options import YAMLOptions

class Options:
    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):  # If value is a dictionary, recursively convert
                setattr(self, key, Options(value))
            else:
                setattr(self, key, value)

    def display_attributes(self):
        attributes = {attr: getattr(self, attr) for attr in self.__dict__.keys()}
        print("Class Attributes:")
        for key, value in attributes.items():
            if isinstance(value, Options):
                print(f"  {key}:")
                value.display_attributes()  # Recursively display nested attributes
            else:
                print(f"  {key}: {value}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="YAML Parameters")
    parser.add_argument("--yaml_path", type=str, help="The path of yaml")
    return parser.parse_args()

def config_settings():
    args = parse_arguments()

    # Load the configuration file
    config = YAMLOptions(args.yaml_path)

    # Display all parameters
    config.display_params()

    options = Options(config._load_yaml())

    return options

