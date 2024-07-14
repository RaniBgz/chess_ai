

'''
An instance of SearchTree has a tree.
This tree is used to explore the moves at a certain width and depth.
Needs functions: build tree (needs to get the top n moves from the ai mode (=width), and a depth parameter)
'''

from utils import chess_state_to_board
from Chess.ChessState import Move
import random

#TODO: Check ending conditions for the tree (does it come from pygame, or from the AI?)
#TODO: Make search more efficient (alpha-beta pruning, etc.)

class Node:
    def __init__(self, move, evaluation, depth, parent=None):
        self.move = move
        self.evaluation = evaluation
        self.depth = depth
        self.parent = parent
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self

    def __repr__(self):
        return f"Node(move={self.move}, evaluation={self.evaluation}, depth={self.depth}, parent={self.parent}, children={self.children})"


class SearchTree:
    pawn_value = 1
    knight_value = 3
    bishop_value = 3
    rook_value = 5
    queen_value = 9

    def __init__(self, ai, width=2, depth=2, min_pruning_depth=1):
        self.ai = ai
        self.width = width
        self.max_depth = depth
        self.root = None
        self.min_pruning_depth = min_pruning_depth
        # Initialize root node


    #Normalize things: always use the same move notation = chess notation, to string.
    # def build_tree(self, gs, base_move=None):
    #     print("In build tree")
    #     current_depth = 0
    #     base_evaluation = self.evaluate_board(gs) #Evaluate current board position, before the AI plays
    #
    #     self.root = Node(move=base_move,
    #                      evaluation=base_evaluation,
    #                      depth=0.0,
    #                      parent=None)
    #     self._build_tree_recursive(gs, self.root, 0)
    #     # for child in self.root.children:
    #     #     print(f"Child: {child}")

    def build_tree(self, gs, base_move=None):
        current_depth = 0
        base_evaluation = self.evaluate_board(gs)

        board = chess_state_to_board(gs)

        self.root = Node(move=base_move, evaluation=base_evaluation, depth=0.0, parent=None)
        self._build_tree_recursive(gs, self.root, 0, float('-inf'), float('inf'), True)


    # def _build_tree_recursive(self, gs, current_node, current_depth):
    #     if current_depth >= self.max_depth:
    #         return
    #
    #     top_moves = self.ai.get_top_n_moves(gs, self.width)
    #
    #     for move in top_moves:
    #         move_obj = Move.fromChessNotation(move, gs.board)
    #         gs.makeMove(move_obj)
    #         move_evaluation = self.evaluate_board(gs)
    #         child_node = Node(move=move, evaluation=move_evaluation, depth=current_depth + 0.5, parent=current_node)
    #         current_node.add_child(child_node)
    #         self._build_tree_recursive(gs, child_node, current_depth + 0.5)
    #         gs.undoMove()

    # def _build_tree_recursive(self, gs, current_node, current_depth, alpha, beta, maximizing_player):
    #     if current_depth >= self.max_depth:
    #         return
    #
    #     top_moves = self.ai.get_top_n_moves(gs, self.width)
    #
    #     for move in top_moves:
    #         move_obj = Move.fromChessNotation(move, gs.board)
    #         gs.makeMove(move_obj)
    #         move_evaluation = self.evaluate_board(gs)
    #         child_node = Node(move=move, evaluation=move_evaluation, depth=current_depth + 0.5, parent=current_node)
    #         current_node.add_child(child_node)
    #
    #         if current_depth >= self.min_pruning_depth:
    #             if maximizing_player:
    #                 alpha = max(alpha, move_evaluation)
    #                 if alpha >= beta:
    #                     gs.undoMove()
    #                     break
    #             else:
    #                 beta = min(beta, move_evaluation)
    #                 if beta <= alpha:
    #                     gs.undoMove()
    #                     break
    #
    #         self._build_tree_recursive(gs, child_node, current_depth + 0.5, alpha, beta, not maximizing_player)
    #         gs.undoMove()

    def _build_tree_recursive(self, gs, current_node, current_depth, alpha, beta, maximizing_player):
        if current_depth >= self.max_depth:
            return

        top_moves = self.ai.get_top_n_moves(gs, self.width)

        for move in top_moves:
            move_obj = Move.fromChessNotation(move, gs.board)
            gs.makeMove(move_obj)
            move_evaluation = self.evaluate_board(gs)
            print("Move evaluation: ", move_evaluation)
            print("Alpha: ", alpha)
            print("Beta: ", beta)
            child_node = Node(move=move, evaluation=move_evaluation, depth=current_depth + 0.5, parent=current_node)
            current_node.add_child(child_node)

            if maximizing_player:
                alpha = max(alpha, move_evaluation)
                if alpha >= beta:
                    gs.undoMove()
                    print("Pruning in maximizing player")
                    break
            else:
                beta = min(beta, move_evaluation)
                if beta <= alpha:
                    gs.undoMove()
                    print("Pruning in minimizing player")
                    break

            self._build_tree_recursive(gs, child_node, current_depth + 0.5, alpha, beta, not maximizing_player)
            gs.undoMove()

    def evaluate_board(self, gs):
        piece_values = {'bp': 1, 'wp': 1, 'bR': 5, 'wR': 5, 'bN': 3, 'wN': 3, 'bB': 3, 'wB':3, 'wQ':9, 'bQ': 9, 'bK': 0, 'wK': 0}
        white_evaluation = 0
        black_evaluation = 0
        # print("Board: ", gs.board)
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

    def get_best_move(self):
        if not self.root:
            return None

        leaf_nodes = self._collect_leaf_nodes(self.root)
        best_evaluation = float('inf')
        best_leaf = None

        for leaf in leaf_nodes:
            if leaf.evaluation < best_evaluation:
                print("Found a better eval: ", leaf.evaluation)
                best_evaluation = leaf.evaluation
                best_leaf = leaf

        if not best_leaf:
            return None

        current_node = best_leaf
        while current_node.parent and current_node.parent != self.root:
            current_node = current_node.parent

        return current_node.move

    def _collect_leaf_nodes(self, node):
        if not node.children:
            return [node]
        leaf_nodes = []
        for child in node.children:
            leaf_nodes.extend(self._collect_leaf_nodes(child))
        return leaf_nodes