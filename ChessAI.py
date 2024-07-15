import numpy as np
import tensorflow as tf
from tensorflow import keras
import chess
import chess.pgn
import time
import os
from utils import chess_state_to_board


#TODO: make training fault-resilient

num_chunks = 1000
batch_size = 4
chunks_to_save = 2

model_folder = './cnn_models_v10'
base_model_name = 'cnn_v10'
model_extension = '.h5'
pretrained_chunks = 0
model_path = os.path.join(model_folder, f'{base_model_name}_{pretrained_chunks}_bs_{batch_size}{model_extension}')


class ChessAI:
    # maps ranks (chess row labels) to row indices
    ranksToRows = {"1": 7, "2": 6, "3": 5, "4": 4, "5": 3, "6": 2, "7": 1, "8": 0}
    # maps files (chess column labels) to column indices
    filesToCols = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
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

        x = keras.layers.Conv2D(32, (3, 3), padding='same')(input_layer)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(32, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(32, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Flatten()(x)
        # x = keras.layers.GlobalAvgPool2D()(x)

        # First dense layer with regularization
        dense1 = keras.layers.Dense(1024, kernel_regularizer=keras.regularizers.l2(0.01))(x)
        dense1 = keras.layers.BatchNormalization()(dense1)
        dense1 = keras.layers.Activation('relu')(dense1)

        # Second dense layer with regularization
        dense2 = keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.01))(dense1)
        dense2 = keras.layers.BatchNormalization()(dense2)
        dense2 = keras.layers.Activation('relu')(dense2)

        # Output layers for starting and ending squares
        start_square = keras.layers.Dense(64, activation='softmax')(dense1)
        end_square = keras.layers.Dense(64, activation='softmax')(dense1)

        # Reshape to 8x8 matrices
        start_square = keras.layers.Reshape((8, 8))(start_square)
        end_square = keras.layers.Reshape((8, 8))(end_square)

        # Create model
        model = keras.Model(inputs=input_layer, outputs=[start_square, end_square])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model Summary: ", model.summary())
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

    def train_on_pgn_chunks_batch(self, num_chunks, batch_size=10, chunks_to_save=10):
        chunks_dir = 'split_pgn_files'
        chunk_files = [os.path.join(chunks_dir, f"chunk_{i}.pgn") for i in range(num_chunks)]
        chunks_processed_since_last_save = 0

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

            chunks_processed_since_last_save += 1
            self.num_chunks_seen += 1

            # Save model after every n chunks
            if chunks_processed_since_last_save >= chunks_to_save:
                save_path = os.path.join(model_folder, f'{base_model_name}_{chunk_index}_bs_{batch_size}{model_extension}')
                self.save_model(save_path)
                print(f"Model saved after processing {chunks_processed_since_last_save} chunks.")
                chunks_processed_since_last_save = 0

        # Save the model one last time if there are remaining unsaved chunks
        if chunks_processed_since_last_save > 0:
            save_path = os.path.join(model_folder, f'{base_model_name}_{chunk_index}_bs_{batch_size}{model_extension}')
            self.save_model(save_path)
            print(f"Model saved after processing the remaining {chunks_processed_since_last_save} chunks.")

    '''Get best move function for 2*8 output'''
    def get_best_move(self, gs):
        board = chess_state_to_board(gs)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            print("No legal moves")
            return None

        input_matrix = self.board_to_input(board)
        input_matrix = np.expand_dims(input_matrix, axis=0)  # Add batch dimension
        start_predictions, end_predictions = self.model.predict(input_matrix, verbose=0)

        start_predictions = start_predictions.reshape(64)
        end_predictions = end_predictions.reshape(64)


        valid_moves = gs.getValidMoves()
        move_scores = []
        for move in valid_moves:
            cn_move = move.getChessNotation()
            start_col = self.filesToCols[cn_move[0]]
            start_row = self.ranksToRows[cn_move[1]]
            end_col = self.filesToCols[cn_move[2]]
            end_row = self.ranksToRows[cn_move[3]]
            start_square_score = start_predictions[start_row * 8 + start_col]
            end_square_score = end_predictions[end_row * 8 + end_col]
            move_score = start_square_score * end_square_score
            # print(f"Move: {cn_move}, Score: {move_score}")
            move_scores.append((move, move_score))

        move_scores.sort(key=lambda x: x[1], reverse=True)

        best_move = move_scores[0][0] if move_scores else None
        best_move_cn = best_move.getChessNotation()

        return best_move_cn

    def get_top_n_moves(self, gs, n=5):
        board = chess_state_to_board(gs)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            print("No legal moves")
            return None

        input_matrix = self.board_to_input(board)
        input_matrix = np.expand_dims(input_matrix, axis=0)  # Add batch dimension
        start_predictions, end_predictions = self.model.predict(input_matrix, verbose=0)

        start_predictions = start_predictions.reshape(64)
        end_predictions = end_predictions.reshape(64)


        valid_moves = gs.getValidMoves()
        move_scores = []
        for move in valid_moves:
            cn_move = move.getChessNotation()
            start_col = self.filesToCols[cn_move[0]]
            start_row = self.ranksToRows[cn_move[1]]
            end_col = self.filesToCols[cn_move[2]]
            end_row = self.ranksToRows[cn_move[3]]
            # print(f"Start col: {start_col}, Start row: {start_row}, End col: {end_col}, End row: {end_row}")
            start_square_score = start_predictions[start_row * 8 + start_col]
            end_square_score = end_predictions[end_row * 8 + end_col]
            move_score = start_square_score * end_square_score
            move_scores.append((move, move_score))

        # move_scores = []
        # for move in legal_moves:
        #     start_square_score = start_predictions[move.from_square]
        #     end_square_score = end_predictions[move.to_square]
        #     move_score = start_square_score * end_square_score
        #     move_scores.append((move, move_score))
        #     # print (f"Move: {move}, Score: {move_score}")

        move_scores.sort(key=lambda x: x[1], reverse=True)

        # Get the top n moves
        top_n_moves = move_scores[:n]

        # Extract just the moves
        top_n_moves = [move[0] for move in top_n_moves]

        top_n_moves_cn = []
        for move in top_n_moves:
            top_n_moves_cn.append(move.getChessNotation())
        # print(f"Top {n} moves: {top_n_moves_cn}")

        return top_n_moves_cn


    def save_model(self, MODEL_PATH):
        if os.path.exists(MODEL_PATH):
            print(f"Model already exists at {MODEL_PATH}. Skipping save.")
        else:
            self.model.save(MODEL_PATH)
            print("Model saved to disk at path: ", MODEL_PATH)

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
    ai.train_on_pgn_chunks_batch(num_chunks=num_chunks, batch_size=batch_size, chunks_to_save=chunks_to_save)  # Train the AI on PGN data chunks

    # ai.save_model(model_path)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time} seconds")

