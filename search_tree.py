

'''
An instance of SearchTree has a tree.
This tree is used to explore the moves at a certain width and depth.
Needs functions: build tree (needs to get the top n moves from the ai mode (=width), and a depth parameter)
'''

from utils import chess_state_to_board
from Chess.ChessState import Move
import random

class SearchTree:
    pawn_value = 1
    knight_value = 3
    bishop_value = 3
    rook_value = 5
    queen_value = 9

    def __init__(self, ai, width=1, depth=2):
        print(f"SearchTree initialized with width {width} and depth {depth}, and ai {ai}")
        self.ai = ai
        self.width = width
        self.depth = depth

    def build_tree(self, gs, depth=None, width=None, parent_move=None):
        print("Building tree")
        if depth is None:
            depth = self.depth
        if width is None:
            width = self.width

        if depth == 0:
            return []

        tree = []

        board = chess_state_to_board(gs)
        top_moves = self.ai.get_top_n_moves(board, width)
        checked_top_moves = []
        for move in top_moves:
            checked_top_moves.append(self.check_move_validity(gs, move))
        print("Checked top moves: ", checked_top_moves)


        for move in checked_top_moves:
            print(f"Simulating move {move}")
            move_obj = Move.getChessNotation(move)
            print("Move object: ", move_obj)
            gs.makeMove(move_obj)
            move_evaluation = self.evaluate_board(gs)
            print(f"Evaluation of board after initial AI's move: {move_evaluation}")
            opponent_moves = self.simulate_opponent_response(gs, width)
            sub_tree = self.build_tree(gs, depth-1, width, parent_move=move if parent_move is None else parent_move)
            tree.append({
                'move': move,
                'evaluation': move_evaluation,
                'opponent_moves': opponent_moves,
                'sub_tree': sub_tree,
                'parent_move': parent_move
            })
            print(f"Tree: {tree}")
            gs.undoMove()

        return tree

    def get_best_direct_move(self, tree):
        best_evaluation = float('inf')
        best_move = None

        def traverse(node):
            nonlocal best_evaluation, best_move
            if 'sub_tree' in node:
                for child in node['sub_tree']:
                    traverse(child)
            else:
                if node['evaluation'] < best_evaluation:
                    best_evaluation = node['evaluation']
                    best_move = node['parent_move']

        for node in tree:
            traverse(node)

        return best_move

    def evaluate_board(self, gs):
        piece_values = {'bp': 1, 'wp': 1, 'bR': 5, 'wR': 5, 'bN': 3, 'wN': 3, 'bB': 3, 'wB':3, 'wQ':9, 'bQ': 9, 'bK': 0, 'wK': 0}
        white_evaluation = 0
        black_evaluation = 0
        print("Board: ", gs.board)
        for row in gs.board:
            for square in row:
                if square != '--':
                    if square[0] == 'w':
                        piece_type = square
                        piece_value = piece_values[piece_type]
                        white_evaluation += piece_value
                    elif square[0] == 'b':
                        piece_type = square
                        piece_value = piece_values[piece_type]
                        black_evaluation += piece_value
        evaluation = white_evaluation - black_evaluation
        return evaluation

    def simulate_opponent_response(self, gs, n):
        print("Simulating opponent responses")
        board = chess_state_to_board(gs)
        top_moves = self.ai.get_top_n_moves(board, n)
        checked_top_moves = []
        for move in top_moves:
            checked_top_moves.append(self.check_move_validity(gs, move))
        print("Top moves for opponent are:", top_moves)
        opponent_evaluations = []

        for move in checked_top_moves:
            move_obj = Move.fromChessNotation(move.uci(), gs.board)
            gs.makeMove(move_obj)
            evaluation = self.evaluate_board(gs)
            print(f"Evaluation of board after opponent's move {move}: {evaluation}")
            # opponent_evaluations.append((move, evaluation))
            opponent_evaluations.append((move_obj, evaluation))
            gs.undoMove()

        return opponent_evaluations


    def check_move_validity(self, gs, input_move):
        validMoves = gs.getValidMoves()
        for move in validMoves:
            if move.startRow == input_move.from_square // 8 and move.startCol == input_move.from_square % 8 and \
                    move.endRow == input_move.to_square // 8 and move.endCol == input_move.to_square % 8:
                print("AI move is valid")
                cn_move = move.getChessNotation()
                return cn_move
        print("AI move is invalid")
        if validMoves:
            print("Valid moves are: ", validMoves)
            random_move = random.choice(validMoves)
            cn_random_move = random_move.getChessNotation()
            return cn_random_move
        else:
            print("No valid moves available. Game over.")
            return None
