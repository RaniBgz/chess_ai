


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
