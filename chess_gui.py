import pygame as p
from constants import cst
from chess_backend import ChessBackend
from config.game_mode import GameMode

class ChessGUI:
    IMAGES = {}

    def __init__(self, game_mode=None):
        self.chess_backend = ChessBackend()
        self.screen = None
        self.clock = None
        self.game_mode = game_mode
        self.game_over = False
        self.player_1 = None
        self.player_2 = None
        self.selected_squares = ()
        self.player_clicks = []
        self.move_made = False
        self.last_move = None
        self.human_turn = False

        self.metrics_saved = False
        self.running = True

        self.initialize()

    def initialize(self):
        screen_title = self.initialize_game_mode()
        self.screen = self.initialize_screen()
        self.clock = self.initialize_clock()
        self.initialize_images()

        self.print_attributes()

    def print_attributes(self):
        print("Screen: ", self.screen)
        print("Clock: ", self.clock)
        print("Game mode: ", self.game_mode)
        print("Move made: ", self.move_made)
        print("Selected squares: ", self.selected_squares)
        print("Player clicks: ", self.player_clicks)
        print("Game over: ", self.game_over)
        print("Player 1: ", self.player_1)
        print("Player 2: ", self.player_2)

    #TODO: Use game mode to know if it's a human turn or ai turn
    def initialize_game_mode(self):
        if self.game_mode == GameMode.HUMAN_VS_AI:
            self.player_1 = "Human"
            self.player_2 = "AI"
            self.human_turn = True
            return "Chess - Human vs AI"
        elif self.game_mode == GameMode.AI_VS_AI:
            self.player_1 = "AI"
            self.player_2 = "AI"
            return "Chess - AI vs AI"
        else:
            print("Invalid game mode, using default: Human vs AI")
            self.player_1 = "Human"
            self.player_2 = "AI"
            self.human_turn = True
            return "Chess - Human vs AI"

    def initialize_screen(self, screen_title="Chess"):
        p.init()
        screen = p.display.set_mode((cst.WIDTH, cst.HEIGHT))
        #TODO: Change the title based on game mode
        p.display.set_caption(screen_title)
        screen.fill(p.Color("white"))
        return screen

    def initialize_clock(self):
        return p.time.Clock()

    def drawText(self, text):
        font = p.font.SysFont("Helvitca", 32, True, False)
        textObject = font.render(text, 0, p.Color('Gray'))
        textLocation = p.Rect(0, 0, cst.WIDTH, cst.HEIGHT).move(cst.WIDTH / 2 - textObject.get_width() / 2,
                                                                cst.HEIGHT / 2 - textObject.get_height() / 2)
        self.screen.blit(textObject, textLocation)
        textObject = font.render(text, 0, p.Color("Black"))
        self.screen.blit(textObject, textLocation.move(2, 2))

    def drawBoard(self):
        global colors
        colors = [cst.light_wood, cst.dark_wood]
        for r in range(cst.DIMENSION):
            for c in range(cst.DIMENSION):
                color = colors[((r + c) % 2)]
                p.draw.rect(self.screen, color, p.Rect(c * cst.SQ_SIZE, r * cst.SQ_SIZE, cst.SQ_SIZE, cst.SQ_SIZE))

    def drawPieces(self, board):
        for r in range(cst.DIMENSION):
            for c in range(cst.DIMENSION):
                piece = board[r][c]
                if piece != "--":
                    self.screen.blit(self.IMAGES[piece], p.Rect(c * cst.SQ_SIZE, r * cst.SQ_SIZE, cst.SQ_SIZE, cst.SQ_SIZE))

    def drawGameState(self, gs, validMoves, sqSelected):
        self.drawBoard(self.screen)  # draw square on the board
        self.highlightSquares(self.screen, gs, validMoves, sqSelected)
        self.drawPieces(self.screen, gs.board)  # draw pieces on top of those squares

    def highlightSquares(self, gs, validMoves, sqSelected):
        if sqSelected != ():
            r, c = sqSelected
            if gs.board[r][c][0] == ('w' if gs.whiteToMove else 'b'):  # sqSelected is a piece that can be move
                s = p.Surface((cst.SQ_SIZE, cst.SQ_SIZE))
                s.set_alpha(100)  # transparent value
                s.fill(p.Color('blue'))
                self.screen.blit(s, (c * cst.SQ_SIZE, r * cst.SQ_SIZE))
                s.fill(p.Color('yellow'))
                for move in validMoves:
                    if move.startRow == r and move.startCol == c:
                        self.screen.blit(s, (move.endCol * cst.SQ_SIZE, move.endRow * cst.SQ_SIZE))

    def animateMove(self, move, board, clock):
        global colors
        dR = move.endRow - move.startRow
        dC = move.endCol - move.startCol
        framesPerSquare = 2  # frames to move one square
        frameCount = (abs(dR) + abs(dC)) * framesPerSquare
        for frame in range(frameCount + 1):
            r, c = (move.startRow + dR * frame / frameCount, move.startCol + dC * frame / frameCount)
            self.drawBoard()
            self.drawPieces( board)
            # erase the piece moved from its ending square
            color = colors[(move.endRow + move.endCol) % 2]
            endSquare = p.Rect(move.endCol * cst.SQ_SIZE, move.endRow * cst.SQ_SIZE, cst.SQ_SIZE, cst.SQ_SIZE)
            p.draw.rect(self.screen, color, endSquare)
            # draw captured piece onto rectangle
            if move.pieceCaptured != '--':
                self.screen.blit(self.IMAGES[move.pieceCaptured], endSquare)
            # draw moving piece
            self.screen.blit(self.IMAGES[move.pieceMoved], p.Rect(c * cst.SQ_SIZE, r * cst.SQ_SIZE, cst.SQ_SIZE, cst.SQ_SIZE))
            p.display.flip()
            clock.tick(60)

    def initialize_images(self):
        pieces = ['wp', 'wR', 'wN', 'wB', 'wK', 'wQ', 'bp', 'bR', 'bN', 'bB', 'bK', 'bQ']
        for piece in pieces:
            self.IMAGES[piece] = p.transform.scale(p.image.load("images/" + piece + ".png"), (cst.SQ_SIZE, cst.SQ_SIZE))


    def run_human_vs_ai(self):
        while self.running:
            if self.human_turn:
                print("Human turn")


    def run_ai_vs_ai(self):
        while self.running:
            pass



if __name__ == "__main__":
    gui = ChessGUI(game_mode=GameMode.HUMAN_VS_AI)
    gui.run_human_vs_ai()