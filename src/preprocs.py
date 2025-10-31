
# ------------------------------------------------------------------------------
#| Aqui se recebe o nome da imagem. Após isso, cria-se uma matriz 3D com os     |
#| valores RGB da iamgem.  Antigamente este código também aplicava um blur,     |
#| mas tiramos por motivos de melhora nos resultados                            |
# ------------------------------------------------------------------------------

import cv2
import numpy as np

def preprocessar_imagem(caminho_imagem, aplicar_blur=True, kernel_blur=(5, 5)):

    """
    Lê uma imagem, converte para o espaço de corRGB e normaliza os canais.

    Esta normalização (0-255 -> 0-1) é específica para que a 
    distância Euclidiana em pesos_grafo.py funcione bem.

    Retorna:
    - np.ndarray: A matriz 3D (imagem) processada RGB normalizada.
                   Retorna None se a imagem não puder ser lida.
    """

    imagem_bgr = cv2.imread(caminho_imagem)

    if imagem_bgr is None:
        print(f"Erro: Não foi possível ler a imagem em '{caminho_imagem}'")
        return None

    imagem_lab_cv2 = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2Lab)

    # Normalizar os valores (0-1)
    imagem_normalizada = imagem_lab_cv2.astype(np.float32) / 255.0

    return imagem_normalizada

# --- Teste local  ---
if __name__ == "__main__":
    
    imagem = preprocessar_imagem("totoro_rebaixado.jpg") 

    if imagem is not None:
        print(f"Shape da Matriz (RGB Processada): {imagem.shape}")
        print(f"Tipo de dados: {imagem.dtype}") 
        
        print(f"Valores Min/Max: {imagem.min():.4f} / {imagem.max():.4f}")