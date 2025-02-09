

'''
An instance of SearchTree has a tree.
This tree is used to explore the moves at a certain width and depth.
Needs functions: build tree (needs to get the top n moves from the ai mode (=width), and a depth parameter)
'''

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
    king_value = 0
    pieces_values = {'bp': pawn_value, 'wp': pawn_value, 'bN': knight_value, 'wN': knight_value, 'bB': bishop_value,
                     'wB': bishop_value, 'bR': rook_value, 'wR': rook_value, 'bQ': queen_value, 'wQ': queen_value,
                     'bK': king_value, 'wK': king_value}

    def __init__(self, ai, width=2, depth=2, min_pruning_depth=1):
        self.ai = ai
        self.width = width
        self.max_depth = depth
        self.root = None
        self.min_pruning_depth = min_pruning_depth
        # Initialize root node

    def get_width(self):
        return self.width

    def get_depth(self):
        return self.max_depth

    def build_tree(self, gs, base_move=None):
        current_depth = 0
        base_evaluation = self.evaluate_board(gs)


        self.root = Node(move=base_move, evaluation=base_evaluation, depth=0.0, parent=None)
        self._build_tree_recursive(gs, self.root, current_depth, float('-inf'), float('inf'), False)



    def _build_tree_recursive(self, gs, current_node, current_depth, alpha, beta, maximizing_player):
        if current_depth >= self.max_depth:
            #Exploring further moves can lead to checkmate positions, need to revert the boolean after tree building
            if gs.checkMate:
                gs.checkMate = False
            if gs.staleMate:
                gs.staleMate = False
            return

        top_moves = self.ai.get_top_n_moves(gs, self.width)

        if top_moves is None or len(top_moves) == 0:
            # No legal moves available, this is either checkmate or stalemate
            if gs.checkMate:
                current_node.evaluation = float('-inf') if maximizing_player else float('inf')
            else:# stalemate
                current_node.evaluation = self.evaluate_board(gs)
            return

        for move in top_moves:
            move_obj = Move.fromChessNotation(move, gs.board)
            # print("Making move: ", move)
            gs.makeMove(move_obj)
            move_evaluation = self.evaluate_board(gs)
            # print("Board right after evaluation: ")
            # gs.print_board()
            child_node = Node(move=move, evaluation=move_evaluation, depth=current_depth + 0.5, parent=current_node)
            current_node.add_child(child_node)

            # if current_depth >= self.min_pruning_depth:
            #     if maximizing_player:
            #         alpha = max(alpha, move_evaluation)
            #         if alpha >= beta:
            #             print("Unmaking move in alpha-beta pruning, max player")
            #             gs.undoMove()
            #             break
            #     else:
            #         beta = min(beta, move_evaluation)
            #         if beta <= alpha:
            #             print("Unmaking move in alpha-beta pruning, min player")
            #             gs.undoMove()
            #             break

            self._build_tree_recursive(gs, child_node, current_depth + 0.5, alpha, beta, not maximizing_player)
            # print("Unmaking move:", move)
            gs.undoMove()

    def evaluate_board(self, gs):
        white_evaluation = 0
        black_evaluation = 0
        # print("Board: ", gs.board)
        for row in gs.board:
            for square in row:
                if square != '--':
                    if square[0] == 'w':
                        piece_type = square
                        piece_value = self.pieces_values[piece_type]
                        white_evaluation += piece_value
                    elif square[0] == 'b':
                        piece_type = square
                        piece_value = self.pieces_values[piece_type]
                        black_evaluation += piece_value
        evaluation = white_evaluation - black_evaluation
        return evaluation

    def get_best_move(self):
        if not self.root or not self.root.children:
            return None

        leaf_nodes = self._collect_leaf_nodes(self.root)
        if not leaf_nodes:
            return None
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