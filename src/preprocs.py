# Arquivo: preprocs.py
# Autor: Pessoa 1
# Objetivo: Carregar a imagem e converter para L*a*b* normalizado
#           para o cálculo de pesos.

import cv2
import numpy as np

def preprocessar_imagem(caminho_imagem, aplicar_blur=True, kernel_blur=(5, 5)):
    """
    Lê uma imagem, converte para o espaço de cor L*a*b*, aplica um blur
    opcional e normaliza os canais.

    Esta normalização (0-255 -> 0-1) é específica para que a 
    distância Euclidiana em pesos_grafo.py (P3) funcione bem.

    Retorna:
    - np.ndarray: A matriz 3D (imagem) processada L*a*b* normalizada.
                   Retorna None se a imagem não puder ser lida.
    """

    # 1. Ler a imagem (BGR)
    imagem_bgr = cv2.imread(caminho_imagem)

    if imagem_bgr is None:
        print(f"Erro: Não foi possível ler a imagem em '{caminho_imagem}'")
        return None

    # 2. Converter para L*a*b*
    # A conversão do OpenCV (uint8) gera L, a, b no intervalo [0, 255]
    imagem_lab_cv2 = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2Lab)

    # 3. Normalizar os valores (0-1)
    
    # O seu código original tinha uma lógica de normalização complexa.
    # Uma forma mais direta de normalizar os 3 canais de [0, 255] para [0, 1]
    # é simplesmente converter para float e dividir por 255.0.
    # Isso alcança o mesmo objetivo para a distância Euclidiana
    # (L_norm = L_orig/255.0, a_norm = a_orig/255.0, b_norm = b_orig/255.0)
    
    imagem_normalizada = imagem_lab_cv2.astype(np.float32) / 255.0
    return imagem_normalizada
'''
    # 4. Aplicar blur opcional
    imagem_final = imagem_normalizada
    if aplicar_blur:
        # O GaussianBlur suaviza a imagem, removendo ruídos.
        imagem_final = cv2.GaussianBlur(imagem_normalizada, kernel_blur, 0)

    # 5. Retornar a matriz L*a*b* normalizada e com blur
    return imagem_final
'''
# --- Teste local (se executado diretamente) ---
if __name__ == "__main__":
    
    # Use "totoro_rebaixado.jpg" ou "balls.png"
    imagem = preprocessar_imagem("totoro_rebaixado.jpg") 

    if imagem is not None:
        print(f"Shape da Matriz (L*a*b* Processada): {imagem.shape}")
        print(f"Tipo de dados: {imagem.dtype}") 
        
        # O intervalo não será mais [0, 1] perfeito por causa do blur
        # e da normalização de 3 canais.
        print(f"Valores Min/Max: {imagem.min():.4f} / {imagem.max():.4f}")