

'''
An instance of SearchTree has a tree.
This tree is used to explore the moves at a certain width and depth.
Needs functions: build tree (needs to get the top n moves from the ai mode (=width), and a depth parameter)
'''

from utils import chess_state_to_board
from Chess.ChessState import Move
import random

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

    def __init__(self, ai, width=2, depth=2):
        self.ai = ai
        self.width = width
        self.max_depth = depth
        self.root = None
        # Initialize root node


    #Normalize things: always use the same move notation = chess notation, to string.
    def build_tree(self, gs, base_move=None):
        print("In build tree")
        current_depth = 0
        base_evaluation = self.evaluate_board(gs) #Evaluate current board position, before the AI plays

        board = chess_state_to_board(gs) #Convert

        self.root = Node(move=base_move,
                         evaluation=base_evaluation,
                         depth=0.0,
                         parent=None)
        # print("Root node: ", self.root)
        # self.root.__repr__()
        # top_moves = self.ai.get_top_n_moves(gs, self.width) #Get top n moves from AI

        self._build_tree_recursive(gs, self.root, 0)

        # for child in self.root.children:
        #     print(f"Child: {child}")


    def _build_tree_recursive(self, gs, current_node, current_depth):
        if current_depth >= self.max_depth:
            return

        top_moves = self.ai.get_top_n_moves(gs, self.width)
        # checked_moves = [self.check_move_validity(gs, move) for move in top_moves]

        for move in top_moves:
            move_obj = Move.fromChessNotation(move, gs.board)
            gs.makeMove(move_obj)
            move_evaluation = self.evaluate_board(gs)
            child_node = Node(move=move, evaluation=move_evaluation, depth=current_depth + 0.5, parent=current_node)
            current_node.add_child(child_node)
            self._build_tree_recursive(gs, child_node, current_depth + 0.5)
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



    # def evaluate_all_moves(self, gs, nodes):
    #     #Iterate through all the depths until the final depth
    #     for i in range(1, 2*self.depth+1):
    #         current_depth = i/2.0
    #         print(current_depth)
    #         for node in nodes:
    #             move_obj = Move.fromChessNotation(str(node.move), gs.board)
    #             gs.makeMove(move_obj)
    #             self.evaluate_moves_at_depth_for_one_parent(gs, current_depth, node, current_depth)


    # def evaluate_moves_at_depth_for_one_parent(self, gs, current_depth, moves, parent_node=None):
    #     #Going throuhg checked top moves, evaluating the board after the move, and creating a node for each move
    #     nodes = []
    #     for move in moves:
    #         move_obj = Move.fromChessNotation(move, gs.board)
    #         gs.makeMove(move_obj)
    #         move_evaluation = self.evaluate_board(gs)
    #         node = Node(move=move, evaluation=move_evaluation, depth=current_depth, parent=parent_node)
    #         parent_node.add_child(node)
    #         print(f"Created node: {node}")
    #         nodes.append(node)
    #         gs.undoMove()
    #     return nodes


    # def get_best_direct_move(self, tree):
    #     best_evaluation = 10000
    #     best_move = None
    #
    #     def traverse(node):
    #         nonlocal best_evaluation, best_move
    #         if 'sub_tree' in node:
    #             for child in node['sub_tree']:
    #                 traverse(child)
    #         else:
    #             if node['evaluation'] < best_evaluation:
    #                 best_evaluation = node['evaluation']
    #                 best_move = node['parent_move']
    #                 print(f"Best move: {best_move} with evaluation {best_evaluation}")
    #
    #     for node in tree:
    #         traverse(node)
    #
    #     return best_move



    # def simulate_opponent_response(self, gs, n):
    #     print("Simulating opponent responses")
    #     board = chess_state_to_board(gs)
    #     top_moves = self.ai.get_top_n_moves(board, n)
    #     checked_top_moves = []
    #     for move in top_moves:
    #         checked_top_moves.append(self.check_move_validity(gs, move))
    #     print("Top moves for opponent are:", top_moves)
    #     opponent_evaluations = []
    #
    #     for move in checked_top_moves:
    #         print("Move in checked moves: ", move)
    #         move_obj = Move.fromChessNotation(str(move), gs.board)
    #         gs.makeMove(move_obj)
    #         evaluation = self.evaluate_board(gs)
    #         print(f"Evaluation of board after opponent's move {move}: {evaluation}")
    #         # opponent_evaluations.append((move, evaluation))
    #         opponent_evaluations.append((move_obj, evaluation))
    #         gs.undoMove()
    #
    #     return opponent_evaluations


    # def check_move_validity(self, gs, input_move):
    #     validMoves = gs.getValidMoves()
    #     cn_validMoves = []
    #     for move in validMoves:
    #         cn_validMoves.append(move.getChessNotation())
    #     for cn_move in cn_validMoves:
    #         if str(cn_move) == str(input_move):
    #             print("AI move is valid")
    #             return str(input_move)
    #     print("AI move is invalid")
    #     if cn_validMoves:
    #         random_move = random.choice(cn_validMoves)
    #         return str(random_move)
    #     else:
    #         print("No valid moves available. Game over.")
    #         return None


        # checked_top_moves = []
        # #Check if moves are valid, replace invalid moves by random moves
        # for move in top_moves:
        #     checked_top_moves.append(self.check_move_validity(gs, move))
        # print("Checked top moves: ", checked_top_moves)

        #First step, evaluate the board after the AI's move at depth = 0.5
        # base_nodes = self.evaluate_moves_at_depth_for_one_parent(gs, current_depth, top_moves,
        #                                                         parent_node=self.root)

        # #First step, evaluate the board after the AI's move at depth = 0.5
        # nodes = self.evaluate_moves_at_depth_for_one_parent(gs, current_depth, checked_top_moves, parent_node=self.root)
        #
        # #Second step: for each node, evaluate the board after the opponent's move at depth = 1
        # for node in nodes:
        #     #Before calling the function again, we need to replay all the moves to be in the same state as before
        #     move_obj = Move.fromChessNotation(str(node.move), gs.board)
        #     gs.makeMove(move_obj)
        #     board = chess_state_to_board(gs)
        #     top_moves2 = self.ai.get_top_n_moves(gs, self.width)
        #     checked_top_moves = []
        #     # Check if moves are valid, replace invalid moves by random moves
        #     for move in top_moves2:
        #         checked_top_moves.append(self.check_move_validity(gs, move))
        #     print("Checked top moves: ", checked_top_moves)
        #     nodes2 = self.evaluate_moves_at_depth_for_one_parent(gs, current_depth+0.5, checked_top_moves, parent_node=node)