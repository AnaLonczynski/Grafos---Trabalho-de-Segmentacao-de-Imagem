# Importa a SUA função do OUTRO arquivo
from preprocs import preprocessar_imagem
import numpy as np  

# Define o nome da imagem que você baixou
NOME_ARQUIVO = "minitotoro.jpeg" # Mude se o nome for outro

print(f"Processando a imagem: {NOME_ARQUIVO}...")

# 1. Teste Básico (com blur)
matriz_processada = preprocessar_imagem(NOME_ARQUIVO)

if matriz_processada is not None:
    print("\n--- Teste 1 (com blur) ---")
    print(f"Tipo da matriz: {type(matriz_processada)}")
    print(f"Formato da matriz (Altura, Largura, Canais): {matriz_processada.shape}")
    print(f"Valor mínimo na matriz: {np.min(matriz_processada):.4f}")
    print(f"Valor máximo na matriz: {np.max(matriz_processada):.4f}")
    # O valor min/max deve estar entre 0.0 e 1.0 (ou muito próximo)

# 2. Teste sem blur
matriz_sem_blur = preprocessar_imagem(NOME_ARQUIVO, aplicar_blur=False)

if matriz_sem_blur is not None:
    print("\n--- Teste 2 (sem blur) ---")
    print(f"Formato da matriz: {matriz_sem_blur.shape}")

print("\nTeste concluído.")