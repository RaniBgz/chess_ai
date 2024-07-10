import numpy as np
import tensorflow as tf
from tensorflow import keras
import chess
import chess.pgn
import time
import io
import os
from stockfish import Stockfish
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

#TODO: make training fault-resilient

pgn_path = './base_pgn_files/lichess_db_standard_rated_2015-08.pgn'

num_chunks = 2
game_numbers = num_chunks * 500
batch_size = 4

model_path = f'./cnn_models_v3/cnn_v3_{game_numbers}_bs_{batch_size}.h5'
# stockfish_path = f'./stockfish/stockfish-ubuntu-x86-64-avx2'

class ChessAI:
    def __init__(self, MODEL_PATH=""):
        print("Model path: ", MODEL_PATH)
        # self.stockfish = Stockfish(path=stockfish_path)
        self.ai_move_scores = []
        self.human_move_scores = []
        self.pretrained_model = False
        if os.path.exists(MODEL_PATH):
            self.model = keras.models.load_model(MODEL_PATH)
            print("Model loaded from disk.")
            # self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            self.model = self.create_model()
            # self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print("Created model")

        # self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)

    def create_model(self):
        input_layer = keras.Input(shape=(8, 8, 12))

        # Convolutional layers with 32 filters
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)

        # Convolutional layers with 64 filters
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)

        # Convolutional layers with 128 filters
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        # x = keras.layers.Flatten()(x)

        # Dense layer
        dense1 = keras.layers.Dense(512, activation='relu')(x)

        # Output layers for starting and ending squares
        start_square = keras.layers.Dense(64, activation='softmax')(dense1)
        end_square = keras.layers.Dense(64, activation='softmax')(dense1)

        # Reshape to 8x8 matrices
        start_square = keras.layers.Reshape((8, 8))(start_square)
        end_square = keras.layers.Reshape((8, 8))(end_square)

        # Create model
        model = keras.Model(inputs=input_layer, outputs=[start_square, end_square])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def board_to_input(self, board):
        piece_chars = 'PRNBQKprnbqk'
        input_matrix = np.zeros((8, 8, 12))

        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                row = i // 8
                col = i % 8
                input_matrix[row, col, piece_chars.index(piece.symbol())] = 1

        return input_matrix

    def train_on_pgn_chunks_batch(self, num_chunks, batch_size=10):
        chunks_dir = 'split_pgn_files'
        chunk_files = [os.path.join(chunks_dir, f"chunk_{i}.pgn") for i in range(num_chunks)]

        for chunk_index, chunk_file in enumerate(chunk_files):
            if not os.path.exists(chunk_file):
                print(f"Chunk file {chunk_file} does not exist.")
                continue

            print(f"Training on chunk {chunk_index + 1}/{num_chunks}")
            inputs, targets_start, targets_end = [], [], []
            with open(chunk_file) as f:
                game_number = 0
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    board = game.board()
                    for move in game.mainline_moves():
                        input_matrix = self.board_to_input(board)

                        # Create target matrices
                        start_square = np.zeros((8, 8))
                        end_square = np.zeros((8, 8))

                        # Set the start and end positions
                        start_row, start_col = move.from_square // 8, move.from_square % 8
                        end_row, end_col = move.to_square // 8, move.to_square % 8
                        start_square[start_row, start_col] = 1
                        end_square[end_row, end_col] = 1

                        inputs.append(input_matrix)
                        targets_start.append(start_square)
                        targets_end.append(end_square)

                        # Train in batches
                        if len(inputs) >= batch_size:
                            self.model.fit(np.array(inputs), [np.array(targets_start), np.array(targets_end)], verbose=0)
                            inputs, targets_start, targets_end = [], [], []

                        board.push(move)

                    # Train remaining samples if any
                    if inputs:
                        self.model.fit(np.array(inputs), [np.array(targets_start), np.array(targets_end)], verbose=0)
                        inputs, targets_start, targets_end = [], [], []

                    game_number += 1
                    print(f"Trained on game {game_number} in chunk {chunk_index + 1}")
            print("Training done on chunk ", chunk_index + 1)
            game_nb = (chunk_index + 1) * 500
            save_path = f'./cnn_models_v3/cnn_v3_{game_nb}.h5'



    '''Get best move function for 2*8 output'''
    def get_best_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        input_matrix = self.board_to_input(board)
        input_matrix = np.expand_dims(input_matrix, axis=0)  # Add batch dimension
        start_predictions, end_predictions = self.model.predict(input_matrix)

        start_predictions = start_predictions.reshape(64)
        end_predictions = end_predictions.reshape(64)

        move_scores = []
        for move in legal_moves:
            start_square_score = start_predictions[move.from_square]
            end_square_score = end_predictions[move.to_square]
            move_score = start_square_score * end_square_score
            move_scores.append((move, move_score))

        move_scores.sort(key=lambda x: x[1], reverse=True)

        best_move = move_scores[0][0] if move_scores else None
        return best_move

    def save_model(self, MODEL_PATH):
        if os.path.exists(MODEL_PATH):
            print(f"Model already exists at {MODEL_PATH}. Skipping save.")
        else:
            self.model.save(MODEL_PATH)
            print("Model saved to disk.")

def chess_state_to_board(gs):
    fen = ""
    for row in gs.board:
        empty = 0
        for piece in row:
            if piece == "--":
                empty += 1
            else:
                if empty > 0:
                    fen += str(empty)
                    empty = 0
                fen += piece[1].lower() if piece[0] == 'b' else piece[1].upper()
        if empty > 0:
            fen += str(empty)
        fen += "/"
    fen = fen[:-1]  # remove last slash
    fen += " w KQkq - 0 1"  # Add default values for now
    return chess.Board(fen)


def split_pgn_file(pgn_file, games_per_chunk=500):
    output_dir = 'split_pgn_files'
    os.makedirs(output_dir, exist_ok=True)

    chunk_count = 0
    with open(pgn_file) as f:
        while True:
            chunk_filename = os.path.join(output_dir, f"chunk_{chunk_count}.pgn")
            with open(chunk_filename, 'w') as chunk_file:
                for _ in range(games_per_chunk):
                    game = chess.pgn.read_game(f)
                    if game is None:
                        return
                    chunk_file.write(str(game) + "\n\n")
            chunk_count += 1


# In your main game loop, when it's AI's turn:
# board = chess_state_to_board(gs)  # Convert your GameState to chess.Board
# best_move = ai.get_best_move(board)
# if best_move:
#     # Convert best_move to your Move class and make the move
#     # For example: gs.makeMove(Move(best_move.from_square, best_move.to_square, gs.board))

if __name__ == "__main__":
    # print("Splitting PGN file into chunks...")
    # split_pgn_file(pgn_path, games_per_chunk=500)


    print("Initializing chessAI")
    ai = ChessAI(MODEL_PATH=model_path)

    print("Training on PGN data chunks...")
    start_time = time.time()
    ai.train_on_pgn_chunks_batch(num_chunks=num_chunks, batch_size=batch_size)  # Train the AI on PGN data chunks

    ai.save_model(model_path)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time} seconds")

    # print("Training on pgn data:", pgn_path)
    # start_time = time.time()
    # # for _ in ai.train_on_pgn(pgn_path, num_games=100):  # Train the AI on PGN data
    # #     pass  # This will execute the generator
    # # ai.train_on_pgn(pgn_path, num_games=10)  # Train the AI on PGN data
    # ai.train_on_pgn(pgn_path, num_games=500)  # Train the AI on PGN data
    #
    # ai.save_model(model_path)
    # end_time = time.time()
    # print(f"Training completed in {end_time - start_time} seconds")

