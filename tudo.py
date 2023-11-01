import sys
from glob import glob
from io import BytesIO
from functools import reduce
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import chess.engine
import re
import time

import sys

import tkinter as tk
import threading

# Define as coordenadas do tabuleiro na tela
left, top, right, bottom = 290, 134, 1124, 970 #Jogo normal
#left, top, right, bottom = 220, 84, 1158, 1023 #Puzzle

# Define a largura e altura de cada célula do tabuleiro com precisão decimal
cell_width = int((right - left) / 8)
cell_height = int((bottom - top) / 8)
cell_mid_x = int(cell_width / 2)
cell_mid_y = int(cell_height / 2)
board_widht = int((right-left))
board_height = int((bottom-top))

# Função que converte uma posição em letra para coordenada numérica
def letter_to_coordinate(letter):
    return (ord(letter) - ord("a"))

# Função que converte uma posição em coordenada numérica para letra
def coordinate_to_letter(coordinate):
    return chr(coordinate + ord("a"))

def coordenada(lance):
    start, end = lance[:2], lance[2:]
    start_row, start_col = int(start[1]) - 1, letter_to_coordinate(start[0])
    end_row, end_col = int(end[1]) - 1, letter_to_coordinate(end[0])

    start_x = left + (cell_width * start_col) + cell_mid_x
    start_y = bottom - (cell_height * start_row) - cell_mid_y
    end_x = left + (cell_width * end_col) + cell_mid_x
    end_y = bottom - (cell_height * end_row) - cell_mid_y

    return (int(start_x), int(start_y)), (int(end_x), int(end_y))

engine = chess.engine.SimpleEngine.popen_uci("C://stockfish//stockfish-windows-x86-64-avx2.exe")

from constants import (
    NN_MODEL_PATH, FEN_CHARS, USE_GRAYSCALE
)
from utils import compressed_fen
from chessboard_image import get_chessboard_tiles

def _chessboard_tiles_img_data(chessboard_img_path, options={}):
    """ Given a file path to a chessboard PNG image, returns a
        size-64 array of 32x32 tiles representing each square of a chessboard
    """
    n_channels = 1 if USE_GRAYSCALE else 3
    tiles = get_chessboard_tiles(chessboard_img_path, use_grayscale=USE_GRAYSCALE)
    img_data_list = []
    for i in range(64):
        buf = BytesIO()
        tiles[i].save(buf, format='PNG')
        img_data = tf.image.decode_image(buf.getvalue(), channels=n_channels)
        img_data = tf.image.convert_image_dtype(img_data, tf.float32)
        img_data = tf.image.resize(img_data, [32, 32])
        img_data_list.append(img_data)
    return img_data_list

def predict_chessboard(chessboard_img_path, options={}):
    """ Given a file path to a chessboard PNG image,
        Returns a FEN string representation of the chessboard
    """
    img_data_list = _chessboard_tiles_img_data(chessboard_img_path, options)
    predictions = []
    for i in range(64):
        # a8, b8 ... g1, h1
        tile_img_data = img_data_list[i]
        (fen_char, probability) = predict_tile(tile_img_data)
        predictions.append((fen_char, probability))
    predicted_fen = compressed_fen(
        '/'.join(
            [''.join(r) for r in np.reshape([p[0] for p in predictions], [8, 8])]
        )
    )
    return predicted_fen

def predict_tile(tile_img_data):
    """ Given the image data of a tile, try to determine what piece
        is on the tile, or if it's blank.

        Returns a tuple of (predicted FEN char, confidence)
    """
    probabilities = list(model.predict(np.array([tile_img_data]))[0])
    max_probability = max(probabilities)
    i = probabilities.index(max_probability)
    return (FEN_CHARS[i], max_probability)

def obter_top_3_lances(fen, cor):
    if cor == "black":
        board = chess.Board(inverte_fen(fen))
    else:
        board = chess.Board(fen)
        
    if cor == "white":
        analysis = engine.analyse(board, chess.engine.Limit(depth=16), multipv=1) 
    else:
        analysis = engine.analyse(board, chess.engine.Limit(depth=16), multipv=1) 
    results = []
    for i, info in enumerate(analysis):
        results.append((info["pv"][0], info["score"].relative.score()))   
    return results

def desenhar(inicio, final, cor):

    def remove_arrow():
        canvas.delete(arrow)
        
    root = tk.Tk()
    root.overrideredirect(True)
    root.lift()
    root.wm_attributes("-topmost", True)
    root.wm_attributes("-disabled", True)
    root.wm_attributes("-transparentcolor", "white")

    canvas = tk.Canvas(root, bg='white', height=1080, width=1728)
    canvas.pack()

    if cor == "black":
        arrow = canvas.create_line(board_widht - (inicio[0] - left) + left, board_height - (inicio[1] - top) + top, board_widht - (final[0] - left) + left, board_height - (final[1] - top) + top, fill='red', width=20, arrow=tk.LAST)
    else:
        arrow = canvas.create_line(inicio[0], inicio[1], final[0], final[1], fill='red', width=20, arrow=tk.LAST)

    root.mainloop()

    root.after(1000, remove_arrow)

def inverte_fen(fen):
    print(fen)
    fen_reverso = fen[::-1]


    print(fen_reverso + " b")
    
    return fen_reverso + " b"

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path/glob to PNG chessboard image(s)")
    parser.add_argument("cor", help="Color to be passed as parameter")
    args = parser.parse_args()  
    
    # Carrega o modelo de rede neural na memória
    model = models.load_model(NN_MODEL_PATH)

    if len(sys.argv) > 1:
        for chessboard_image_path in sorted(glob(args.image_path)):
            start_time = time.time()
            fen = predict_chessboard(chessboard_image_path, args)
            print(f"Tempo para identificar o tabuleiro: {time.time() - start_time:.2f} segundos")
            start_time = time.time()
            lances = obter_top_3_lances(fen, args.cor)
            print(f"Tempo para obter os melhores lances: {time.time() - start_time:.2f} segundos")
            for lance, pontuacao in lances:
                start_time = time.time()
                lance = lance.uci()
                starttt, enddd = coordenada(lance)
                print(f"Tempo para calcular posição do lance: {time.time() - start_time:.2f} segundos")
                # Cria um novo thread para desenhar a seta na tela
                thread = threading.Thread(target=desenhar, args=(starttt, enddd, args.cor))
                thread.start()