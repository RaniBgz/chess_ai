import os


class ConfigManager:
    def __init__(self, path):
        self.path = path
        pass


    def load_config(self):
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as file:
                return yaml.safe_load(file)
        else:
            return {'model_path': './chess_model.h5', 'use_checkpoint': False}

    # Save configuration
    def save_config(config):
        with open(CONFIG_PATH, 'w') as file:
            yaml.safe_dump(config, file)
