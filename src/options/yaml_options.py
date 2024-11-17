import yaml

class YAMLOptions:
    """
    A class for managing machine learning parameters using YAML files.
    """
    def __init__(self, yaml_file):
        """
        Initialize the parameter manager.
        :param yaml_file: Path to the YAML configuration file.
        """
        self.yaml_file = yaml_file
        self.params = self._load_yaml()

    def _load_yaml(self):
        """
        Load parameters from a YAML file.
        :return: A dictionary containing the parameters.
        """
        with open(self.yaml_file, "r") as f:
            return yaml.safe_load(f)

    def save_yaml(self, output_file=None):
        """
        Save parameters to a YAML file.
        :param output_file: Path to save the updated parameters. Defaults to overwriting the original file.
        """
        output_file = output_file or self.yaml_file
        with open(output_file, "w") as f:
            yaml.safe_dump(self.params, f)
        print(f"Parameters saved to {output_file}.")

    def get_section(self, section):
        """
        Get a specific section of the parameters.
        :param section: The name of the section (e.g., "train", "test", "general").
        :return: A dictionary of parameters for the specified section, or None if the section does not exist.
        """
        return self.params.get(section, None)

    def get_multiple_sections(self, sections):
        """
        Get multiple sections of the parameters.
        :param sections: A list of section names (e.g., ["train", "test"]).
        :return: A dictionary containing the requested sections.
        """
        return {section: self.params.get(section, {}) for section in sections}

    def update_section(self, section, updates):
        """
        Update parameters for a specific section.
        :param section: The name of the section (e.g., "train", "test", "general").
        :param updates: A dictionary containing the parameters to update.
        """
        if section not in self.params:
            self.params[section] = {}
        self.params[section].update(updates)

    def display_section(self, section):
        """
        Display the parameters for a specific section.
        :param section: The name of the section (e.g., "train", "test", "general").
        """
        section_params = self.get_section(section)
        if section_params:
            print(f"Parameters for section '{section}':")
            print(yaml.dump(section_params, default_flow_style=False))
        else:
            print(f"Section '{section}' does not exist.")

    def display_params(self):
        """
        Display all the parameters
        """
        if self.params:
            print(f"Parameters:")
            print(yaml.dump(self.params, default_flow_style=False))
        else:
            print(f"Parameters do not exist.")


# Example: Load YAML configuration and manage sections
if __name__ == "__main__":
    # Load the configuration file
    config = YAMLOptions("./config/test.yaml")

    # Display all parameters
    print("=== All Parameters ===")
    config.display_params()

    # Retrieve and display specific sections
    # print("\n=== Train and Test Parameters ===")
    # train_and_test_params = config.get_multiple_sections(["train", "test"])
    # print(yaml.dump(train_and_test_params, default_flow_style=False))

    # Save updated parameters to a new YAML file
    # config.save_yaml("updated_config.yaml")

