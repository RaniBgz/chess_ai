

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

    def __init__(self, ai, width=2, depth=2):
        self.ai = ai
        self.depth = depth
        self.width = width
        # Initialize root node


    #Normalize things: always use the same move notation = chess notation, to string.
    def build_tree(self, gs, base_move=None):
        print("In build tree")
        current_depth = 0.0
        base_evaluation = self.evaluate_board(gs) #Evaluate current board position, before the AI plays

        board = chess_state_to_board(gs) #Convert

        self.root = Node(move=base_move,
                         evaluation=base_evaluation,
                         depth=0.0,
                         parent=None)
        print("Root node: ", self.root)
        # self.root.__repr__()
        top_moves = self.ai.get_top_n_moves(board, self.width) #Get top n moves from AI

        checked_top_moves = []
        #Check if moves are valid, replace invalid moves by random moves
        for move in top_moves:
            checked_top_moves.append(self.check_move_validity(gs, move))
        print("Checked top moves: ", checked_top_moves)

        current_depth = current_depth+0.5

        #First step, evaluate the board after the AI's move at depth = 0.5
        base_nodes = self.evaluate_moves_at_depth_for_one_parent(gs, current_depth, checked_top_moves,
                                                                parent_node=self.root)


        #First step, evaluate the board after the AI's move at depth = 0.5
        nodes = self.evaluate_moves_at_depth_for_one_parent(gs, current_depth, checked_top_moves, parent_node=self.root)

        #Second step: for each node, evaluate the board after the opponent's move at depth = 1
        for node in nodes:
            #Before calling the function again, we need to replay all the moves to be in the same state as before
            move_obj = Move.fromChessNotation(str(node.move), gs.board)
            gs.makeMove(move_obj)
            board = chess_state_to_board(gs)
            top_moves2 = self.ai.get_top_n_moves(board, self.width)
            checked_top_moves = []
            # Check if moves are valid, replace invalid moves by random moves
            for move in top_moves2:
                checked_top_moves.append(self.check_move_validity(gs, move))
            print("Checked top moves: ", checked_top_moves)
            nodes2 = self.evaluate_moves_at_depth_for_one_parent(gs, current_depth+0.5, checked_top_moves, parent_node=node)




        #TODO: Predict top n moves
        #TODO: Check if moves are valid
        #TODO: For each move, evaluate board after move
        #TODO: For each move, create Node object
        #TODO: For each move, Add child node to parent node

        pass

    def evaluate_all_moves(self, gs, nodes):
        #Iterate through all the depths until the final depth
        for i in range(1, 2*self.depth+1):
            current_depth = i/2.0
            print(current_depth)
            for node in nodes:
                move_obj = Move.fromChessNotation(str(node.move), gs.board)
                gs.makeMove(move_obj)
                self.evaluate_moves_at_depth_for_one_parent(gs, current_depth, node, current_depth)


    def evaluate_moves_at_depth_for_one_parent(self, gs, current_depth, moves, parent_node=None):
        #Going throuhg checked top moves, evaluating the board after the move, and creating a node for each move
        nodes = []
        for move in moves:
            move_obj = Move.fromChessNotation(move, gs.board)
            gs.makeMove(move_obj)
            move_evaluation = self.evaluate_board(gs)
            node = Node(move=move, evaluation=move_evaluation, depth=current_depth, parent=parent_node)
            parent_node.add_child(node)
            print(f"Created node: {node}")
            nodes.append(node)
            gs.undoMove()
        return nodes

            # #TODO: Then we increment depth by 0.5, and do this for all the nodes
            # opponent_moves = self.simulate_opponent_response(gs, self.width)
            # print("Building sub tree")
            # sub_tree = self.build_tree(gs, base_move=move)
            # self.root.add_child(Node(move=move, evaluation=move_evaluation, depth=1))
            # print(f"Tree: {self.root}")
            # gs.undoMove()


    # def build_tree(self, gs, depth=None, width=None, parent_move=None):
    #     print("Building tree with depth ", depth, " and width ", width, " and parent move ", parent_move)
    #     if depth is None:
    #         depth = depth
    #     if width is None:
    #         width = width
    #
    #     if depth == 0:
    #         return []
    #
    #     tree = []
    #
    #     board = chess_state_to_board(gs)
    #     top_moves = self.ai.get_top_n_moves(board, width)
    #     checked_top_moves = []
    #     for move in top_moves:
    #         checked_top_moves.append(self.check_move_validity(gs, move))
    #     print("Checked top moves: ", checked_top_moves)
    #
    #
    #     for move in checked_top_moves:
    #         print(f"Simulating move {move}")
    #         move_obj = Move.fromChessNotation(str(move), gs.board)
    #         print("Move object: ", move_obj)
    #         gs.makeMove(move_obj)
    #         move_evaluation = self.evaluate_board(gs)
    #         print(f"Evaluation of board after initial AI's move: {move_evaluation}")
    #         opponent_moves = self.simulate_opponent_response(gs, width)
    #         print("Building sub tree")
    #         sub_tree = self.build_tree(gs, depth-1, width, parent_move=move if parent_move is None else parent_move)
    #         tree.append({
    #             'move': move,
    #             'evaluation': move_evaluation,
    #             'opponent_moves': opponent_moves,
    #             'sub_tree': sub_tree,
    #             'parent_move': parent_move
    #         })
    #         print(f"Tree: {tree}")
    #         gs.undoMove()
    #
    #     return tree

    def get_best_direct_move(self, tree):
        best_evaluation = 10000
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
                    print(f"Best move: {best_move} with evaluation {best_evaluation}")

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
            print("Move in checked moves: ", move)
            move_obj = Move.fromChessNotation(str(move), gs.board)
            gs.makeMove(move_obj)
            evaluation = self.evaluate_board(gs)
            print(f"Evaluation of board after opponent's move {move}: {evaluation}")
            # opponent_evaluations.append((move, evaluation))
            opponent_evaluations.append((move_obj, evaluation))
            gs.undoMove()

        return opponent_evaluations


    def check_move_validity(self, gs, input_move):
        validMoves = gs.getValidMoves()
        cn_validMoves = []
        for move in validMoves:
            cn_validMoves.append(move.getChessNotation())
        for cn_move in cn_validMoves:
            if str(cn_move) == str(input_move):
                print("AI move is valid")
                return str(input_move)
        print("AI move is invalid")
        if cn_validMoves:
            random_move = random.choice(cn_validMoves)
            return str(random_move)
        else:
            print("No valid moves available. Game over.")
            return None


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
