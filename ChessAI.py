import numpy as np
import tensorflow as tf
from tensorflow import keras
import chess
import chess.pgn
import time
import io
import os
import datetime


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

print(tf.sysconfig.get_build_info())
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("GPUs available:")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPUs found")

pgn_path = './lichess_db_standard_rated_2015-08.pgn'
# pgn_path = './Carlsen.pgn'
# PGN_PATH = './test.pgn'

model_path = './cnn_v3_lichess_500.h5'


class ChessAI:
    def __init__(self, MODEL_PATH=""):
        print("Model path: ", MODEL_PATH)
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

    # def create_model(self):
    #     model = keras.Sequential([
    #         keras.layers.Dense(128, activation='relu', input_shape=(64 * 12,)),
    #         keras.layers.Dense(128, activation='relu'),
    #         keras.layers.Dense(64, activation='softmax')
    #     ])
    #
    #     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #     return model
    #

    # def board_to_input(self, board):
    #     piece_chars = 'PRNBQKprnbqk'
    #     input_vector = np.zeros(64 * 12)
    #     for i in range(64):
    #         piece = board.piece_at(i)
    #         if piece:
    #             input_vector[i * 12 + piece_chars.index(piece.symbol())] = 1
    #     return input_vector

    # def train_on_pgn(self, pgn_file, num_games=1000):
    #     print("inside train on pgn")
    #     is_trained = False
    #     with open(pgn_file) as f:
    #         print(f"TRAINING FILE {pgn_file} HAS BEEN OPENED")
    #         for game_number in range(num_games):
    #             print(f"Training on game {game_number+1}")
    #             game = chess.pgn.read_game(f)
    #             # print(game)
    #             if game is None:
    #                 break
    #             board = game.board()
    #             for move in game.mainline_moves():
    #                 input_vector = self.board_to_input(board)
    #                 target = np.zeros(64)
    #                 target[move.from_square] = 1
    #                 self.model.fit(np.array([input_vector]), np.array([target]), verbose=0)
    #                 move_made = move.__str__()
    #                 board.push(move)
    #                 yield (game_number+1), is_trained, move_made, False  # False indicates game not ended
    #             yield (game_number+1), is_trained, move_made, True  # True indicates game ended
    #     is_trained = True
    #     yield (game_number+1), is_trained, move_made, True
    #     # self.save_model(model_path)

    # def get_best_move(self, board):
    #     legal_moves = list(board.legal_moves)
    #     if not legal_moves:
    #         return None
    #
    #     input_vector = self.board_to_input(board)
    #     predictions = self.model.predict(np.array([input_vector]))[0]
    #     print("Predictions:", predictions)
    #
    #     move_scores = [(move, predictions[move.from_square]) for move in legal_moves]
    #     move_scores.sort(key=lambda x: x[1], reverse=True)
    #
    #     for move, score in move_scores:
    #         print(f"Considering move {move} with score {score}")
    #
    #     best_move = move_scores[0][0] if move_scores else None
    #     return best_move

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

    def split_pgn_file(self, pgn_file, games_per_chunk=500):
        chunk_count = 0
        with open(pgn_file) as f:
            while True:
                chunk_filename = f"{pgn_file}_chunk_{chunk_count}.pgn"
                with open(chunk_filename, 'w') as chunk_file:
                    for _ in range(games_per_chunk):
                        game = chess.pgn.read_game(f)
                        if game is None:
                            return
                        chunk_file.write(str(game) + "\n\n")
                chunk_count += 1

    def train_on_pgn_chunks(self, pgn_file, num_chunks):
        chunk_files = [f"{pgn_file}_chunk_{i}.pgn" for i in range(num_chunks)]

        for chunk_index, chunk_file in enumerate(chunk_files):
            if not os.path.exists(chunk_file):
                print(f"Chunk file {chunk_file} does not exist.")
                continue

            print(f"Training on chunk {chunk_index + 1}/{num_chunks}")
            with open(chunk_file) as f:
                game_number = 0
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    board = game.board()
                    for move in game.mainline_moves():
                        input_matrix = self.board_to_input(board)
                        input_matrix = np.expand_dims(input_matrix, axis=0)  # Add batch dimension

                        # Create target matrices
                        start_square = np.zeros((8, 8))
                        end_square = np.zeros((8, 8))

                        # Set the start and end positions
                        start_row, start_col = move.from_square // 8, move.from_square % 8
                        end_row, end_col = move.to_square // 8, move.to_square % 8
                        start_square[start_row, start_col] = 1
                        end_square[end_row, end_col] = 1

                        start_square = np.expand_dims(start_square, axis=0)  # Add batch dimension
                        end_square = np.expand_dims(end_square, axis=0)  # Add batch dimension
                        self.model.fit(input_matrix, [start_square, end_square], verbose=0)
                        board.push(move)
                    game_number += 1
                    print(f"Trained on game {game_number} in chunk {chunk_index + 1}")


    def train_on_pgn(self, pgn_file, num_games=1000):
        is_trained = False
        with open(pgn_file) as f:
            for game_number in range(num_games):
                print(f"Training on game {game_number + 1}")
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                board = game.board()
                for move in game.mainline_moves():
                    input_matrix = self.board_to_input(board)
                    input_matrix = np.expand_dims(input_matrix, axis=0)  # Add batch dimension

                    # Create target matrices
                    start_square = np.zeros((8, 8))
                    end_square = np.zeros((8, 8))

                    # Set the start and end positions
                    start_row, start_col = move.from_square // 8, move.from_square % 8
                    end_row, end_col = move.to_square // 8, move.to_square % 8
                    start_square[start_row, start_col] = 1
                    end_square[end_row, end_col] = 1

                    start_square = np.expand_dims(start_square, axis=0)  # Add batch dimension
                    end_square = np.expand_dims(end_square, axis=0)  # Add batch dimension
                    self.model.fit(input_matrix, [start_square, end_square], verbose=0)
                    board.push(move)

        is_trained = True
        print("Training completed")

    # def train_on_pgn(self, pgn_file, num_games=1000):
    #     is_trained = False
    #     with open(pgn_file) as f:
    #         for game_number in range(num_games):
    #             print(f"Training on game {game_number + 1}")
    #             game = chess.pgn.read_game(f)
    #             if game is None:
    #                 break
    #             board = game.board()
    #             for move in game.mainline_moves():
    #                 input_matrix = self.board_to_input(board)
    #                 input_matrix = np.expand_dims(input_matrix, axis=0)  # Add batch dimension
    #
    #                 # Create target matrices
    #                 start_square = np.zeros((8, 8))
    #                 end_square = np.zeros((8, 8))
    #
    #                 # Set the start and end positions
    #                 start_row, start_col = move.from_square // 8, move.from_square % 8
    #                 end_row, end_col = move.to_square // 8, move.to_square % 8
    #                 start_square[start_row, start_col] = 1
    #                 end_square[end_row, end_col] = 1
    #
    #                 start_square = np.expand_dims(start_square, axis=0)  # Add batch dimension
    #                 end_square = np.expand_dims(end_square, axis=0)  # Add batch dimension
    #                 self.model.fit(input_matrix, [start_square, end_square], verbose=0)
    #                 board.push(move)
    #                 yield (game_number + 1), is_trained, move.__str__(), False  # False indicates game not ended
    #             yield (game_number + 1), is_trained, move.__str__(), True  # True indicates game ended
    #     is_trained = True
    #     yield (game_number + 1), is_trained, move.__str__(), True

    # def train_on_pgn(self, pgn_file, num_games=1000):
    #     is_trained = False
    #     with open(pgn_file) as f:
    #         print(f"TRAINING FILE {pgn_file} HAS BEEN OPENED")
    #         for game_number in range(num_games):
    #             print(f"Training on game {game_number + 1}")
    #             game = chess.pgn.read_game(f)
    #             if game is None:
    #                 break
    #             board = game.board()
    #             for move in game.mainline_moves():
    #                 input_matrix = self.board_to_input(board)
    #                 input_matrix = np.expand_dims(input_matrix, axis=0)  # Add batch dimension
    #
    #                 # Creating the target vector
    #                 target = np.zeros((1, 64 * 64))
    #                 target_index = move.from_square * 64 + move.to_square
    #                 target[0, target_index] = 1
    #
    #                 print("Model: ", self.model.summary())
    #                 self.model.fit(input_matrix, target, verbose=0)
    #                 move_made = move.__str__()
    #                 board.push(move)
    #                 yield (game_number + 1), is_trained, move_made, False  # False indicates game not ended
    #             yield (game_number + 1), is_trained, move_made, True  # True indicates game ended
    #     is_trained = True
    #     yield (game_number + 1), is_trained, move_made, True

    #PGN CNN
    # def train_on_pgn(self, pgn_file, num_games=1000):
    #     is_trained = False
    #     with open(pgn_file) as f:
    #         print(f"TRAINING FILE {pgn_file} HAS BEEN OPENED")
    #         for game_number in range(num_games):
    #             print(f"Training on game {game_number + 1}")
    #             game = chess.pgn.read_game(f)
    #             if game is None:
    #                 break
    #             board = game.board()
    #             for move in game.mainline_moves():
    #                 input_matrix = self.board_to_input(board)
    #                 input_matrix = np.expand_dims(input_matrix, axis=0)  # Add batch dimension
    #                 target = np.zeros((1, 64))
    #                 target[0, move.from_square] = 1
    #                 self.model.fit(input_matrix, target, verbose=0)
    #                 move_made = move.__str__()
    #                 board.push(move)
    #                 yield (game_number + 1), is_trained, move_made, False  # False indicates game not ended
    #             yield (game_number + 1), is_trained, move_made, True  # True indicates game ended
    #     is_trained = True
    #     yield (game_number + 1), is_trained, move_made, True

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

    '''Get best move function for 64x64 output'''
    # def get_best_move(self, board):
    #     legal_moves = list(board.legal_moves)
    #     if not legal_moves:
    #         return None
    #
    #     input_matrix = self.board_to_input(board)
    #     input_matrix = np.expand_dims(input_matrix, axis=0)  # Add batch dimension
    #     predictions = self.model.predict(input_matrix)[0]
    #
    #     move_scores = [(move, predictions[move.from_square * 64 + move.to_square]) for move in legal_moves]
    #     move_scores.sort(key=lambda x: x[1], reverse=True)
    #
    #     for move, score in move_scores:
    #         print(f"Considering move {move} with score {score}")
    #
    #     best_move = move_scores[0][0] if move_scores else None
    #     return best_move

    '''Get best move function for 64 output'''
    #Get best move CNN
    # def get_best_move(self, board):
    #     legal_moves = list(board.legal_moves)
    #     if not legal_moves:
    #         return None
    #
    #     input_matrix = self.board_to_input(board)
    #     input_matrix = np.expand_dims(input_matrix, axis=0)  # Add batch dimension
    #     predictions = self.model.predict(input_matrix)[0]
    #
    #     move_scores = [(move, predictions[move.from_square]) for move in legal_moves]
    #     move_scores.sort(key=lambda x: x[1], reverse=True)
    #
    #     for move, score in move_scores:
    #         print(f"Considering move {move} with score {score}")
    #
    #     best_move = move_scores[0][0] if move_scores else None
    #     return best_move

    def save_model(self,MODEL_PATH):
        self.model.save(MODEL_PATH)
        print("Model saved to disk.")



# Function to convert ChessState to chess.Board
# def chess_state_to_board(gs):
#     fen = ""
#     for row in gs.board:
#         empty = 0
#         for piece in row:
#             if piece == "--":
#                 empty += 1
#             else:
#                 if empty > 0:
#                     fen += str(empty)
#                     empty = 0
#                 fen += piece[1].lower() if piece[0] == 'b' else piece[1].upper()
#         if empty > 0:
#             fen += str(empty)
#             fen += str(empty)
#         fen += "/"
#     fen = fen[:-1]  # remove last slash
#     fen += " w KQkq - 0 1"  # Add default values for now
#     return chess.Board(fen)

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

# In your main game loop, when it's AI's turn:
# board = chess_state_to_board(gs)  # Convert your GameState to chess.Board
# best_move = ai.get_best_move(board)
# if best_move:
#     # Convert best_move to your Move class and make the move
#     # For example: gs.makeMove(Move(best_move.from_square, best_move.to_square, gs.board))

if __name__ == "__main__":
    print("Initializing chessAI")
    ai = ChessAI(MODEL_PATH=model_path)

    print("AI from main:", ai)
    print("Model from main:", ai.model)

    print("Training on pgn data:", pgn_path)
    start_time = time.time()
    # for _ in ai.train_on_pgn(pgn_path, num_games=100):  # Train the AI on PGN data
    #     pass  # This will execute the generator
    # ai.train_on_pgn(pgn_path, num_games=10)  # Train the AI on PGN data
    ai.train_on_pgn(pgn_path, num_games=500)  # Train the AI on PGN data

    ai.save_model(model_path)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time} seconds")

