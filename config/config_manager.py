import os
import yaml
from ChessAI import ChessAI

#TODO: check paths in config for the model, depending on calling script.

# search_tree = SearchTree(ai, width=TREE_WIDTH, depth=TREE_DEPTH)


'''
    Class to manage the configuration file.
    Only cares about getting the arguments from the config file.
    Does not build objects.
'''
class ConfigManager:
    def __init__(self, path):
        self.path = path
        self.config = self.load_config()


    def load_config(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as file:
                return yaml.safe_load(file)
        else:
            print("Config file not found, creating new one. Pleace check the path.")

    def save_config(config):
        with open(CONFIG_PATH, 'w') as file:
            yaml.safe_dump(config, file)

    def get_ai_model_path(self):
        if 'model_path' in self.config:
            return self.config['model_path']
        else:
            return None

    def get_tree_width(self):
        if 'tree_width' in self.config:
            return self.config['tree_width']
        else:
            return None

    def get_tree_depth(self):
        if 'tree_depth' in self.config:
            return self.config['tree_depth']
        else:
            return None


if __name__ == "__main__":
    CONFIG_PATH = '../config.yaml'
    config_manager = ConfigManager(CONFIG_PATH)
    print(config_manager.config)