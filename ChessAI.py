import numpy as np
import tensorflow as tf
from tensorflow import keras
import chess
import chess.pgn
import time
import io
import os
from stockfish import Stockfish


pgn_path = './base_pgn_files/lichess_db_standard_rated_2015-08.pgn'

num_chunks = 2
game_numbers = num_chunks * 500
batch_size = 2

model_path = f'./cnn_models_v3/cnn_v3_{game_numbers}.h5'
stockfish_path = f'./stockfish/stockfish-ubuntu-x86-64-avx2'

class ChessAI:
    def __init__(self, MODEL_PATH=""):
        print("Model path: ", MODEL_PATH)
        self.stockfish = Stockfish(path=stockfish_path)
        self.ai_move_scores = []
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

    def train_on_pgn_chunks(self, num_chunks):
        chunks_dir = 'split_pgn_files'
        chunk_files = [os.path.join(chunks_dir, f"chunk_{i}.pgn") for i in range(num_chunks)]

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

    def save_model(self,MODEL_PATH):
        self.model.save(MODEL_PATH)
        print("Model saved to disk.")

    def score_move(self, gs, ai_move, n_top_moves=20):
        top_moves = self.get_top_moves(gs, n_top_moves)
        ai_move_index = -1
        ai_move_accuracy = 0.0
        print("Top moves length: ", len(top_moves))
        print("Ai move: ", ai_move)
        for i in range(0, len(top_moves)):
            print(f"Current top move: {top_moves[i]['Move']}")
            print(f"Current ai move: {ai_move}")
            if top_moves[i]['Move'] == ai_move:
                ai_move_index = i
                break
        if ai_move_index == -1:
            print(f"AI move {ai_move} not found in top {n_top_moves} moves")
        else:
            accuracy_step = 100/n_top_moves
            ai_move_accuracy = 100.0 - (ai_move_index * accuracy_step)
        self.ai_move_scores.append([ai_move, ai_move_accuracy])
        print(f"ai_move_scores: {self.ai_move_scores}")


    def compute_average_accuracy(self):
        total_accuracy = 0
        for move in self.ai_move_scores:
            total_accuracy += move[1]
        average_accuracy = total_accuracy / len(self.ai_move_scores)
        print("Average accuracy: ", average_accuracy)
        return average_accuracy


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

    #Unused at the moment: static evaluation
    def evaluate_move(self, gs):
        fen_position = board_to_fen(gs)
        print("Fen position: ", fen_position)
        self.stockfish.set_fen_position(fen_position)
        return self.stockfish.get_evaluation()


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

def board_to_fen(gs):
    # Define the mappings for row to rank and column to file
    rowsToRanks = {7: '1', 6: '2', 5: '3', 4: '4', 3: '5', 2: '6', 1: '7', 0: '8'}
    colsToFiles = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}

    board = gs.board
    fen = ""
    for row in board:
        empty_count = 0
        for square in row:
            if square == "--":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen += str(empty_count)
                    empty_count = 0
                fen += square[1].lower() if square[0] == 'b' else square[1].upper()
        if empty_count > 0:
            fen += str(empty_count)
        fen += "/"
    fen = fen[:-1]  # Remove the last slash

    # Add active color
    fen += " w" if gs.whiteToMove else " b"

    # Castling rights
    castling_rights = ""
    if gs.currentCastlingRight.wks:
        castling_rights += "K"
    if gs.currentCastlingRight.wqs:
        castling_rights += "Q"
    if gs.currentCastlingRight.bks:
        castling_rights += "k"
    if gs.currentCastlingRight.bqs:
        castling_rights += "q"
    fen += " " + castling_rights if castling_rights else " -"

    # En passant target square
    if gs.enpassantPossible:
        row, col = gs.enpassantPossible
        fen += " " + rowsToRanks[row] + colsToFiles[col]
    else:
        fen += " -"

    # Halfmove clock and fullmove number (set to 0 and 1 for simplicity)
    fen += " 0 1"

    return fen


# def board_to_fen(gs):
#     board = gs.board
#     fen = ""
#     for row in board:
#         empty_count = 0
#         for square in row:
#             if square == "--":
#                 empty_count += 1
#             else:
#                 if empty_count > 0:
#                     fen += str(empty_count)
#                     empty_count = 0
#                 fen += square[1].lower() if square[0] == 'b' else square[1].upper()
#         if empty_count > 0:
#             fen += str(empty_count)
#         fen += "/"
#     fen = fen[:-1]  # Remove the last slash
#
#     # Add active color
#     fen += " b" if gs.whiteToMove else " w"
#
#     # Castling rights
#     castling_rights = ""
#     if gs.currentCastlingRight.wks:
#         castling_rights += "K"
#     if gs.currentCastlingRight.wqs:
#         castling_rights += "Q"
#     if gs.currentCastlingRight.bks:
#         castling_rights += "k"
#     if gs.currentCastlingRight.bqs:
#         castling_rights += "q"
#     fen += " " + castling_rights if castling_rights else " -"
#
#     # En passant target square
#     fen += " " + ("-" if not gs.enpassantPossible else f"{gs.enpassantPossible[0]}{gs.enpassantPossible[1]}")
#
#     # Halfmove clock and fullmove number (set to 0 and 1 for simplicity)
#     fen += " 0 1"
#
#     return fen


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

