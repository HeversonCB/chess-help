import tkinter as tk
import subprocess
import os
import pyautogui
import threading
import time


left, top, right, bottom = 290, 134, 1124, 970 #Jogo normal
#left, top, right, bottom = 220, 84, 1158, 1023 #Puzzle
processo = None
LOOP_RUNNING = False


def iniciar_loop():
    global LOOP_RUNNING
    while LOOP_RUNNING:
        start_time = time.time()
        iniciar_tudo()
        print(f"Tempo para iniciar tudo: {time.time() - start_time:.2f} segundos")
        time.sleep(4)
        parar_tudo()


def iniciar_tudo():
    start_time = time.time()
    if os.path.exists("chess.png"):
        os.remove("chess.png")
    screenshot = pyautogui.screenshot(region=(left, top, right-left, bottom-top))
    screenshot.save("chess.png")
    print(f"Tempo para salvar imagem: {time.time() - start_time:.2f} segundos")
    global processo
    processo = subprocess.Popen(['python', 'tudo.py', 'chess.png', cor.get()])


def parar_tudo():
    global processo
    if processo is not None:
        processo.terminate()
        processo = None


def iniciar_loop_btn():
    global LOOP_RUNNING
    if not LOOP_RUNNING:
        LOOP_RUNNING = True
        threading.Thread(target=iniciar_loop, daemon=True).start()


def parar_loop_btn():
    global LOOP_RUNNING
    if LOOP_RUNNING:
        LOOP_RUNNING = False


# Cria a janela principal
janela = tk.Tk()
janela.geometry("200x200")

cor = tk.StringVar()
cor.set("white")

janela.wm_attributes("-topmost", True)  # Mantém sempre no topo

# Cria os botões
botao_iniciar = tk.Button(janela, text="Iniciar loop", command=iniciar_loop_btn)
botao_iniciar.pack(pady=10)

botao_parar = tk.Button(janela, text="Parar loop", command=parar_loop_btn)
botao_parar.pack(pady=10)

rb_preto = tk.Radiobutton(janela, text="Preto", variable=cor, value="black")
rb_preto.pack(pady=10)

rb_branco = tk.Radiobutton(janela, text="Branco", variable=cor, value="white")
rb_branco.pack(pady=10)

# Inicia a janela principal
janela.mainloop()