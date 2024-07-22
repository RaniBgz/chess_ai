from config.config_manager import ConfigManager
from ChessAI import ChessAI
from metrics import Metrics
from search_tree import SearchTree

import pygame as p

class ChessBackend:
    CONFIG_PATH = './config.yaml'

    #Config manager, ai model, metrics, tree, game state
    def __init__(self):
        self.config_manager = ConfigManager(self.CONFIG_PATH)
        self.ai = None
        self.metrics = None
        self.search_tree = None

    def initialize_ai_model(self, ai_model_path=None):
        ai = None
        if ai_model_path:
            ai = ChessAI(MODEL_PATH=ai_model_path)
        else:
            ai = ChessAI()
        return ai

    def initialize_metrics(self, ai_model_path=None):
        return Metrics(model_name=ai_model_path)

    def initialize_search_tree(self):
        tree_width = self.config_manager.get_tree_width()
        tree_depth = self.config_manager.get_tree_depth()
        return SearchTree(self.ai, width=tree_width, depth=tree_depth)

    def initialize(self):
        ai_model_path = self.config_manager.get_ai_model_path()
        self.ai = self.initialize_ai_model(ai_model_path)
        self.metrics = self.initialize_metrics(ai_model_path)
        self.search_tree = self.initialize_search_tree()
        pass

    def run(self):
        pass