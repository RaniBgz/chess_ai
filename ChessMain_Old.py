import multiprocessing
import random
import time
import threading
import pygame as p
import yaml
import os
from Chess import ChessState
from ChessAI import ChessAI
from utils import chess_state_to_board
from metrics import Metrics
from search_tree import SearchTree

WIDTH = HEIGHT = 512
DIMENSION = 8  # 8*8 board
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 60
IMAGES = {}
BOARDER_SIZE = 40
LABEL_FONT_SIZE = 20
NUM_GAMES_TRAIN = 100
PGN_PATH = './base_pgn_files/lichess_db_standard_rated_2015-08.pgn'
# PGN_PATH = './test.pgn'
#PGN_PATH = './lichess_db_standard_rated_2018-08.pgn.crdownload'
CONFIG_PATH = './config.yaml'

TREE_WIDTH = 3
TREE_DEPTH = 3
MIN_PRUNING_DEPTH = 2

# Load configuration
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as file:
            return yaml.safe_load(file)
    else:
        return {'model_path': './chess_model.h5', 'use_checkpoint': False}

# Save configuration
def save_config(config):
    with open(CONFIG_PATH, 'w') as file:
        yaml.safe_dump(config, file)

config = load_config()

# Initialize global dictionary of images
def loadImages():
    pieces = ['wp', 'wR', 'wN', 'wB', 'wK', 'wQ', 'bp', 'bR', 'bN', 'bB', 'bK', 'bQ']
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load("images/" + piece + ".png"), (SQ_SIZE, SQ_SIZE))

# MAIN, to handle user input and update graphics
def main():
    p.init()
    screen = p.display.set_mode((WIDTH, HEIGHT))
    clock = p.time.Clock()
    screen.fill(p.Color("white"))
    gs = ChessState.GameState()
    validMoves = gs.getValidMoves()
    moveMade = False  # flag var for when a move is made
    animate = False  # flag variable for when we should use animate a move
    loadImages()
    running = True
    sqSelected = ()
    playerClicks = []
    gameOver = False
    playerOne = True  # If a human is playing white, else False
    playerTwo = False  # If a human is playing black, else False
    last_human_move = None
    n_top_moves = 10
    total_ai_moves = 0
    replaced_moves = 0
    winner = "Tie"


    metrics_saved = False

    evaluations = []
    model_name = config['model_path'].split('/')[-1]
    metrics = Metrics(model_name=model_name)

    if os.path.exists(config['model_path']):
        ai = ChessAI(MODEL_PATH=config['model_path'])
    else:
        ai = ChessAI()

    search_tree = SearchTree(ai, width=TREE_WIDTH, depth=TREE_DEPTH)

    while running:
        humanTurn = (gs.whiteToMove and playerOne) or (not gs.whiteToMove and playerTwo)
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False
            # mouse handle
            elif e.type == p.MOUSEBUTTONDOWN:
                if not gameOver and humanTurn:
                    location = p.mouse.get_pos()  # (x, y) location of mouse
                    col = location[0] // SQ_SIZE
                    row = location[1] // SQ_SIZE
                    if sqSelected == (row, col):  # user click same square twice
                        sqSelected = ()  # deselect
                        playerClicks = []
                    else:
                        sqSelected = (row, col)
                        playerClicks.append(sqSelected)  # for both first and second clicks
                    if len(playerClicks) == 2:  # after second click
                        move = ChessState.Move(playerClicks[0], playerClicks[1], gs.board)
                        for i in range(len(validMoves)):
                            if move == validMoves[i]:
                                last_human_move = str(validMoves[i].getChessNotation())
                                gs.makeMove(validMoves[i])
                                moveMade = True
                                animate = True
                                sqSelected = ()  # reset user click
                                playerClicks = []
                                #Human move: Engine is too slow to score all moves
                        if not moveMade:
                            playerClicks = [sqSelected]
            # key handler
            elif e.type == p.KEYDOWN:
                if e.key == p.K_z:  # undo when z is pressed
                    gs.undoMove() if playerTwo else gs.undoLastTwoMoves()      
                    moveMade = True
                    animate = False
                if e.key == p.K_r:  # reset game when 'r' is pressed
                    gs = ChessState.GameState()
                    validMoves = gs.getValidMoves()
                    sqSelected = ()
                    playerClicks = []
                    moveMade = False
                    animate = False
                    gameOver = False

        # AI move finder
        if not gameOver and not humanTurn:
            '''These two lines to use tree search or not.'''
            # ai_move = ai_move_without_tree_search(ai, gs)
            ai_move = ai_move_with_tree_search(search_tree, gs, last_human_move=last_human_move)
            print("Human move: ", gs.moveLog[-1].getChessNotation() if gs.moveLog else "None", "AI move: ", ai_move)
            if ai_move:
                print("AI making move: ", ai_move)
                metrics.score_move(gs, ai_move, n_top_moves=n_top_moves)
                move = ChessState.Move.fromChessNotation(ai_move, gs.board)
                gs.makeMove(move)
                moveMade = True
                animate = True
                total_ai_moves += 1
            elif ai_move is None:
                print("AI couldn't make a valid move. Choosing a random move.")
                validMoves = gs.getValidMoves()
                if validMoves:
                    random_move = random.choice(validMoves)
                    cn_random_move = random_move.getChessNotation()
                    metrics.score_move(gs, cn_random_move, n_top_moves=n_top_moves)
                    gs.makeMove(random_move)
                    moveMade = True
                    animate = True
                    replaced_moves += 1
                    total_ai_moves += 1
                else:
                    print("No valid moves available. Game over.")
                    gameOver = True
        if moveMade:
            if animate:
                animateMove(gs.moveLog[-1], screen, gs.board, clock)
            validMoves = gs.getValidMoves()
            moveMade = False
            animate = False

        drawGameState(screen, gs, validMoves, sqSelected)

        if gs.checkMate:
            gameOver = True
            if gs.whiteToMove:
                drawText(screen, 'Black wins by checkmate')
            else:
                drawText(screen, 'White wins by checkmate')
        elif gs.staleMate:
            gameOver = True
            drawText(screen, 'Stalemate')

        #Storing metrics at the end of the game
        if gameOver:
            if not metrics_saved:
                threading.Thread(target=plot_accuracy, args=(metrics,)).start()
                winner = 'White' if gs.whiteToMove else 'Black'
                threading.Thread(target=save_game_summary, args=(metrics, winner, total_ai_moves, replaced_moves)).start()
                metrics_saved = True
        clock.tick(MAX_FPS)
        p.display.flip()

def ai_move_without_tree_search(ai, gs):
    ai_move = ai.get_best_move(gs)
    return ai_move

def ai_move_with_tree_search(search_tree, gs, last_human_move=None):
    search_tree.build_tree(gs, last_human_move)
    ai_move = search_tree.get_best_move()
    return ai_move

def plot_accuracy(metrics):
    print("Inside plot accuracy in main")
    metrics.save_plot()

def save_game_summary(metrics, winner, total_ai_moves, replaced_moves):
    print("Inside plot accuracy in main")
    metrics.save_game_summary(winner, total_ai_moves, replaced_moves)



def highlightSquares(screen, gs, validMoves, sqSelected):
    if sqSelected != ():
        r, c = sqSelected
        if gs.board[r][c][0] == ('w' if gs.whiteToMove else 'b'):  # sqSelected is a piece that can be move
            s = p.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100)  # transparent value
            s.fill(p.Color('blue'))
            screen.blit(s, (c * SQ_SIZE, r * SQ_SIZE))
            s.fill(p.Color('yellow'))
            for move in validMoves:
                if move.startRow == r and move.startCol == c:
                    screen.blit(s, (move.endCol * SQ_SIZE, move.endRow * SQ_SIZE))

def drawGameState(screen, gs, validMoves, sqSelected):
    drawBoard(screen)  # draw square on the board
    highlightSquares(screen, gs, validMoves, sqSelected)
    drawPieces(screen, gs.board)  # draw pieces on top of those squares

def drawBoard(screen):
    global colors
    colors = [p.Color("white"), p.Color("gray")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[((r + c) % 2)]
            p.draw.rect(screen, color, p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def drawPieces(screen, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != "--":
                screen.blit(IMAGES[piece], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def animateMove(move, screen, board, clock):
    global colors
    dR = move.endRow - move.startRow
    dC = move.endCol - move.startCol
    framesPerSquare = 2  # frames to move one square
    frameCount = (abs(dR) + abs(dC)) * framesPerSquare
    for frame in range(frameCount + 1):
        r, c = (move.startRow + dR * frame / frameCount, move.startCol + dC * frame / frameCount)
        drawBoard(screen)
        drawPieces(screen, board)
        # erase the piece moved from its ending square
        color = colors[(move.endRow + move.endCol) % 2]
        endSquare = p.Rect(move.endCol * SQ_SIZE, move.endRow * SQ_SIZE, SQ_SIZE, SQ_SIZE)
        p.draw.rect(screen, color, endSquare)
        # draw captured piece onto rectangle
        if move.pieceCaptured != '--':
            screen.blit(IMAGES[move.pieceCaptured], endSquare)
        # draw moving piece
        screen.blit(IMAGES[move.pieceMoved], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
        p.display.flip()
        clock.tick(60)

def drawText(screen, text):
    font = p.font.SysFont("Helvitca", 32, True, False)
    textObject = font.render(text, 0, p.Color('Gray'))
    textLocation = p.Rect(0, 0, WIDTH, HEIGHT).move(WIDTH / 2 - textObject.get_width() / 2,
                                                    HEIGHT / 2 - textObject.get_height() / 2)
    screen.blit(textObject, textLocation)
    textObject = font.render(text, 0, p.Color("Black"))
    screen.blit(textObject, textLocation.move(2, 2))

if __name__ == "__main__":
    main()
