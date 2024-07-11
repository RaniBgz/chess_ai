import numpy as np
import tensorflow as tf
from tensorflow import keras
import chess
import chess.pgn
import time
import os

#TODO: make training fault-resilient

num_chunks = 100
batch_size = 1

model_folder = './cnn_models_v4'
base_model_name = 'cnn_v4'
model_extension = '.h5'
pretrained_chunks = 1
model_path = os.path.join(model_folder, f'{base_model_name}_{pretrained_chunks}_bs_{batch_size}{model_extension}')

class ChessAI:
    def __init__(self, pretrained=False, MODEL_PATH="", num_chunks_seen=0):
        print("Model path: ", MODEL_PATH)
        self.pretrained = pretrained
        self.num_chunks_seen = num_chunks_seen
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
        input_layer = keras.Input(shape=(12, 8, 8))

        # Convolutional layers with 32 filters
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)

        # Convolutional layers with 64 filters
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)

        # Convolutional layers with 128 filters
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        # x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Flatten()(x)

        # Dense layer
        dense1 = keras.layers.Dense(1024, activation='relu')(x)
        dense1 = keras.layers.Dropout(0.5)(dense1)

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
        input_matrix = np.zeros((12, 8, 8))
        # print("Board is ", board)
        # print("Input matrix is ", input_matrix)

        for i in range(64):
            piece = board.piece_at(i)
            # print("Piece is ", piece)
            if piece:
                row = i // 8
                col = i % 8
                # print(f"Row: {row}, Col: {col}, Piece: {piece.symbol()}, Piece index: {piece_chars.index(piece.symbol())}")
                input_matrix[piece_chars.index(piece.symbol()), row, col] = 1

        return input_matrix

    def train_on_pgn_chunks_batch(self, num_chunks, batch_size=10):
        chunks_dir = 'split_pgn_files'
        chunk_files = [os.path.join(chunks_dir, f"chunk_{i}.pgn") for i in range(num_chunks)]

        for chunk_index, chunk_file in enumerate(chunk_files):
            if chunk_index < self.num_chunks_seen:
                continue  # Skip chunks that have already been seen

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
                    print(f"Trained on game {game_number} in chunk {chunk_index}")
            print("Training done on chunk ", chunk_index)
            save_path = os.path.join(f'.{model_folder,}',
                                      f'{base_model_name}_{chunk_index}_bs_{batch_size}{model_extension}')
            self.save_model(save_path)

    # def train_on_pgn_chunks_batch(self, num_chunks, batch_size=10):
    #     chunks_dir = 'split_pgn_files'
    #     chunk_files = [os.path.join(chunks_dir, f"chunk_{i}.pgn") for i in range(num_chunks)]
    #
    #     for chunk_index, chunk_file in enumerate(chunk_files):
    #         if not os.path.exists(chunk_file):
    #             print(f"Chunk file {chunk_file} does not exist.")
    #             continue
    #
    #         print(f"Training on chunk {chunk_index + 1}/{num_chunks}")
    #         inputs, targets_start, targets_end = [], [], []
    #         with open(chunk_file) as f:
    #             game_number = 0
    #             while True:
    #                 game = chess.pgn.read_game(f)
    #                 if game is None:
    #                     break
    #                 board = game.board()
    #                 for move in game.mainline_moves():
    #                     input_matrix = self.board_to_input(board)
    #
    #                     # Create target matrices
    #                     start_square = np.zeros((8, 8))
    #                     end_square = np.zeros((8, 8))
    #
    #                     # Set the start and end positions
    #                     start_row, start_col = move.from_square // 8, move.from_square % 8
    #                     end_row, end_col = move.to_square // 8, move.to_square % 8
    #                     start_square[start_row, start_col] = 1
    #                     end_square[end_row, end_col] = 1
    #
    #                     inputs.append(input_matrix)
    #                     targets_start.append(start_square)
    #                     targets_end.append(end_square)
    #
    #                     # Train in batches
    #                     if len(inputs) >= batch_size:
    #                         self.model.fit(np.array(inputs), [np.array(targets_start), np.array(targets_end)], verbose=0)
    #                         inputs, targets_start, targets_end = [], [], []
    #
    #                     board.push(move)
    #
    #                 # Train remaining samples if any
    #                 if inputs:
    #                     self.model.fit(np.array(inputs), [np.array(targets_start), np.array(targets_end)], verbose=0)
    #                     inputs, targets_start, targets_end = [], [], []
    #
    #                 game_number += 1
    #                 print(f"Trained on game {game_number} in chunk {chunk_index + 1}")
    #         print("Training done on chunk ", chunk_index + 1)
    #         save_path = f'./cnn_models_v3/cnn_v3_{chunk_index}_bs_{batch_size}.h5'
    #         self.save_model(save_path)



    '''Get best move function for 2*8 output'''
    def get_best_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        input_matrix = self.board_to_input(board)
        print(f"Input matrix {input_matrix}")
        input_matrix = np.expand_dims(input_matrix, axis=0)  # Add batch dimension
        start_predictions, end_predictions = self.model.predict(input_matrix)
        print(f"Start predictions {start_predictions}")
        print(f"End predictions {end_predictions}")

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
            print("Model saved to disk at path: ", MODEL_PATH)

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

if __name__ == "__main__":
    pretrained = os.path.exists(model_path)
    print("Pretrained is ", pretrained)
    num_chunks_seen = 0
    if pretrained:
        num_chunks_seen = int(model_path.split('_')[-3])
        print("Num chunks seen is ", num_chunks_seen)

    print("Initializing chessAI")
    print("Initializing chessAI")
    ai = ChessAI(MODEL_PATH=model_path, pretrained=pretrained, num_chunks_seen=num_chunks_seen)

    print("Training on PGN data chunks...")
    start_time = time.time()
    ai.train_on_pgn_chunks_batch(num_chunks=num_chunks, batch_size=batch_size)  # Train the AI on PGN data chunks

    # ai.save_model(model_path)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time} seconds")

