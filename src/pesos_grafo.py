# Arquivo: grafo_pesos.py
import numpy as np

def calcular_pesos_arestas(matriz_imagem, arestas):
    """
    Calcula o peso de cada aresta no grafo.
    O peso é a distância Euclidiana entre as cores (L*a*b*)
    dos dois pixels que a aresta conecta.

    Parâmetros:
    - matriz_imagem (np.ndarray): A matriz 3D (Altura x Largura x 3) 
                                  vinda do preprocessamento.
    - arestas (list): A lista de tuplas (u, v) vinda do grafo 
                      básico.

    Retorna:
    - list: Uma lista de tuplas no formato (peso, u, v), 
            pronta para ser usada por um algoritmo de AGM (MST).
    """
    print(f"Iniciando cálculo de pesos para {len(arestas)} arestas...")

    # 1. Obter dimensões da imagem
    altura, largura, _ = matriz_imagem.shape
    
    # 2. Criar a lista de saída
    arestas_com_pesos = []
    
    # 3. Percorrer cada aresta (u, v)
    for u, v in arestas:
        
        # 4. Converter os IDs 'u' e 'v' de volta para coordenadas (linha, coluna)
        
        # Coordenadas do pixel 'u'
        linha_u = u // largura
        coluna_u = u % largura
        
        # Coordenadas do pixel 'v'
        linha_v = v // largura
        coluna_v = v % largura
        
        # 5. Obter os vetores de cor L*a*b* para cada pixel
        pixel_u_cor = matriz_imagem[linha_u, coluna_u]
        pixel_v_cor = matriz_imagem[linha_v, coluna_v]
        
        # 6. Calcular a distância Euclidiana (Peso)
        peso = np.linalg.norm(pixel_u_cor - pixel_v_cor)
        
        # 7. Adicionar à lista no formato (peso, u, v)
        arestas_com_pesos.append((peso, u, v))
        
    print("Cálculo de pesos concluído.")
    
    return arestas_com_pesos
