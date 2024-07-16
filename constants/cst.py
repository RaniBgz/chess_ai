import pygame as p


#Screen constants
WIDTH = HEIGHT = 512
DIMENSION = 8  # 8*8 board
SQ_SIZE = HEIGHT // DIMENSION

MAX_FPS = 60

# Define custom colors for a wooden-looking chess board
light_wood = p.Color(205, 170, 125)  # Light wood color
dark_wood = p.Color(139, 69, 19)     # Dark wood color