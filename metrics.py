'''
The metrics class is used to store the accuracy of the moves made by the agent.
'''
import chess
import chess.pgn
import time
from stockfish import Stockfish
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from utils import board_to_fen

stockfish_path = f'./stockfish/stockfish-ubuntu-x86-64-avx2'

class Metrics:
    def __init__(self):
        self.stockfish = Stockfish(path=stockfish_path)
        self.ai_move_scores = []
        self.ai_accuracies = []

        # Set up the plotting figure and axis
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'r-')  # Initialize the line for AI accuracies
        self.ax.set_xlim(0, 100)  # Set x-axis limit (adjust as needed)
        self.ax.set_ylim(0, 100)  # Set y-axis limit (0 to 100% accuracy)
        self.ax.set_xlabel('Move Number')
        self.ax.set_ylabel('Accuracy')
        self.ax.set_title('AI Move Accuracy Over Time')

        # Initialize the animation
        # self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=1000, blit=True)

    def save_plot(self, filename='./metrics/ai_accuracy_plot.png'):
        # Save the plot to a file
        self.fig.savefig(filename)
        print(f"Plot saved as {filename}")

    def update_plot(self):
        # Update the plot with new data
        xdata = list(range(1, len(self.ai_accuracies) + 1))
        ydata = self.ai_accuracies
        self.line.set_data(xdata, ydata)
        self.ax.set_xlim(0, max(100, len(self.ai_accuracies) + 1))  # Dynamically adjust x-axis limit
        return self.line,


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
        self.update_plot()
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
