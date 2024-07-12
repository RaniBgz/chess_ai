'''
The metrics class is used to store the accuracy of the moves made by the agent.
'''
import chess
import chess.pgn
import os
import json
from stockfish import Stockfish
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from utils import board_to_fen
from datetime import datetime

stockfish_path = f'./stockfish/stockfish-ubuntu-x86-64-avx2'
metrics_path = './metrics'
plot_path = './metrics/ai_accuracy.png'

class Metrics:
    def __init__(self):
        self.stockfish = Stockfish(path=stockfish_path)
        self.ai_move_scores = []
        self.ai_accuracies = []
        self.plot_dir = self.initialize_plot_dir()
        # Initialize the animation
        # self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=1000, blit=True)

    def initialize_plot_dir(self):
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.plot_dir = os.path.join(metrics_path, timestamp)
        os.makedirs(self.plot_dir, exist_ok=True)
        self.plot_path = os.path.join(self.plot_dir, 'ai_accuracy_plot.png')


    def plot_ai_accuracy(self):
        if not self.ai_move_scores:
            return
        move_scores = [move[1] for move in self.ai_move_scores]
        plt.plot(move_scores)
        plt.xlabel('Move Number')
        plt.ylabel('Accuracy')
        plt.title('AI Move Accuracy')
        # plt.show()

    def save_plot(self):
        if not self.ai_move_scores:
            return
        move_scores = [move[1] for move in self.ai_move_scores]
        plt.plot(move_scores)
        plt.xlabel('Move Number')
        plt.ylabel('Accuracy')
        plt.title('AI Move Accuracy')
        plt.savefig(self.plot_path)

    '''
    Scores the move that was just made by comparing the move to stockfish's top n moves (max 20)
    Depending on if the move is human or AI, the move is scored and added to the respective list
    In the future: make this even more abstract (black/white) to support AI vs AI play
    '''
    def score_move(self, gs, move, humanTurn = False, n_top_moves=20):
        top_moves = self.get_top_moves(gs, n_top_moves)
        move_index = -1
        move_accuracy = 0.0
        for i in range(0, len(top_moves)):
            print(f"Top move {i}: {top_moves[i]['Move']}")
            if top_moves[i]['Move'] == move:
                move_index = i
                break
        if move_index == -1:
            print(f"Move {move} not found in top {n_top_moves} moves, accuracy is 0")
        else:
            accuracy_step = 100/n_top_moves
            move_accuracy = 100.0 - (move_index * accuracy_step)
        # if humanTurn:
        #     self.human_move_scores.append([move, move_accuracy])
        self.ai_move_scores.append([move, move_accuracy])
        self.ai_accuracies.append(move_accuracy)
        print(f"AI move scores: {self.ai_move_scores}")
        self.compute_average_accuracy()
        # print(f"Human move scores: {self.human_move_scores}")


    def compute_average_accuracy(self):
        total_human_accuracy = 0
        total_ai_accuracy = 0

        # #Compute human
        # for move in self.human_move_scores:
        #     total_human_accuracy += move[1]
        # average_human_accuracy = total_human_accuracy / len(self.human_move_scores)
        # print("Average Human accuracy: ", average_human_accuracy)

        for move in self.ai_move_scores:
            total_ai_accuracy += move[1]
        average_ai_accuracy = total_ai_accuracy / len(self.ai_move_scores)
        print("Average AI accuracy: ", average_ai_accuracy)
        return average_ai_accuracy


    def get_top_moves(self, gs, n):
        fen_position = board_to_fen(gs)
        self.stockfish.set_fen_position(fen_position)
        print(f"Fen position: {fen_position}")
        try:
            top_moves = self.stockfish.get_top_moves(n)
        except:
            print("Error: Stockfish could not get top moves")
            return []
        return top_moves or []
        # print("Top moves: ", top_moves)
