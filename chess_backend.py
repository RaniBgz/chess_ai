from config.config_manager import ConfigManager
from ChessAI import ChessAI
from metrics import Metrics
from search_tree import SearchTree
from Chess import ChessState
import threading
import random

class ChessBackend:
    CONFIG_PATH = './config.yaml'

    #Config manager, ai model, metrics, tree, game state
    def __init__(self):
        self.config_manager = ConfigManager(self.CONFIG_PATH)
        self.ai = None
        self.metrics = None
        self.search_tree = None
        self.game_state = ChessState.GameState()
        self.valid_moves = self.game_state.getValidMoves()

        self.initialize()

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

    def reset_game_state(self):
        self.game_state = ChessState.GameState()
        self.valid_moves = self.game_state.getValidMoves()

    def make_human_move(self, player_clicks):
        move = ChessState.Move(player_clicks[0], player_clicks[1], self.game_state.board)
        for i in range(len(self.valid_moves)):
            if move == self.valid_moves[i]:
                self.game_state.makeMove(self.valid_moves[i])
                self.valid_moves = self.game_state.getValidMoves()
                return move
        return None

    def get_best_ai_move(self, last_move=None):
        ai_move = None
        if self.search_tree:
            self.search_tree.build_tree(self.game_state, last_move)
            ai_move = self.search_tree.get_best_move()
            self.metrics.score_move(self.game_state, ai_move, humanTurn=False)
        else:
            ai_move = self.ai.get_best_move(self.game_state)
        return ai_move

    def make_ai_move(self, ai_move):
        if ai_move: #Valid AI move was found
            #TODO: Add metrics
            move = ChessState.Move.fromChessNotation(ai_move, self.game_state.board)
            self.game_state.makeMove(move)
            self.valid_moves = self.game_state.getValidMoves()
            return True
        else: #No valid AI move was found
            self.valid_moves = self.game_state.getValidMoves()
            if self.valid_moves:
                random_move = random.choice(self.valid_moves)
                self.game_state.makeMove(random_move)
                return True
            else: #No valid moves in this position, game is over
                return False

    def save_metrics(self, winner, total_ai_moves=0, replaced_moves=0):
        print("Winner: ", winner)
        threading.Thread(target=self.metrics.save_plot).start()
        threading.Thread(target=self.metrics.save_game_summary, args=(winner,)).start()
        print("Metrics saved")


    def run(self):
        pass