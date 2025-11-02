
# ------------------------------------------------------------------------------
#| Aqui se recebe a matriz gerada da imagem e o array com as arestas.           |
#| O objetivo é calcular os pesos do grafo por meio da distância euclidiana     |
#| das cores RGB e retorna as arestas com peso                                  |
# ------------------------------------------------------------------------------

import numpy as np

def calcular_pesos_arestas(matriz_imagem, arestas, barra_progresso=None):
    """
    Parâmetros:
    - matriz_imagem (np.ndarray): A matriz 3D (Altura x Largura x 3) 
    - arestas (list): A lista de tuplas (u, v) vinda do grafo básico.

    Retorna:
    - list: Uma lista de tuplas no formato (peso, u, v), 
            pronta para ser usada por um algoritmo de AGM (MST).
    """

    altura, largura, _ = matriz_imagem.shape
    arestas_com_pesos = []
    
    # Percorrer cada aresta (u, v)
    for u, v in arestas:
        
        # Converter os IDs 'u' e 'v' de volta para coordenadas (linha, coluna)
        
        # Coordenadas do pixel 'u'
        linha_u = u // largura
        coluna_u = u % largura
        
        # Coordenadas do pixel 'v'
        linha_v = v // largura
        coluna_v = v % largura
        
        # Obter os vetores de cor RGB para cada pixel
        pixel_u_cor = matriz_imagem[linha_u, coluna_u]
        pixel_v_cor = matriz_imagem[linha_v, coluna_v]
        
        # Calcular a distância Euclidiana (Peso)
        peso = np.linalg.norm(pixel_u_cor - pixel_v_cor)
        
        # Adicionar à lista no formato (w, u, v)
        arestas_com_pesos.append((peso, u, v))

        if barra_progresso:
            barra_progresso.update(1)
        
    return arestas_com_pesos