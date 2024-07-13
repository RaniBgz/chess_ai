''' Utils functions for the chess game, such as converting game chess to board,'''

import os
import chess

pgn_path = './base_pgn_files/lichess_db_standard_rated_2015-08.pgn'


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



if __name__ == "__main__":
    pass
    # print("Splitting PGN file into chunks...")
    # split_pgn_file(pgn_path, games_per_chunk=500)
