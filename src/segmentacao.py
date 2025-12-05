import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

class UnionFind:
    """
    Estrutura para gerenciar os grupos de pixels (segmentos).
    """
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.num_components = n

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])  
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            # Union by Size
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i
            
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            self.num_components -= 1
            return True
        return False

def segmentar_arborescencia(
    pais: Dict[int, Tuple[int, float]], # ID do nó(filho), tupla(ID do pai, peso)
    limiar: float, # Valor de Corte
    num_pixels: int, 
    dimensoes: Tuple[int, int]
) -> np.ndarray:

    print(f"Iniciando segmentação. Limiar: {limiar}")
    
    uf = UnionFind(num_pixels)
    arestas_mantidas = 0
    
    # Itera sobre o dicionário de pais gerado pelo Edmonds
    for filho, (pai, peso) in pais.items():

        if pai is None or pai == -1:
            continue
            
        # Se o peso da conexão dirigida for baixo, une.
        # Se for alto (> limiar), corta a aresta.
        if peso <= limiar:
           
            if uf.union(filho, pai):
                arestas_mantidas += 1

    print(f" Segmentação concluída.")
    print(f" - Arestas mantidas: {arestas_mantidas}")
    print(f" - Segmentos resultantes: {uf.num_components}")

    # Mapear os componentes do UnionFind para uma matriz 2D
    altura, largura = dimensoes
    rotulos_map = np.zeros(dimensoes, dtype=int)
    
    mapa_ids_finais = {}
    contador_ids = 0

    for i in range(num_pixels):
        raiz = uf.find(i)
        
        if raiz not in mapa_ids_finais:
            mapa_ids_finais[raiz] = contador_ids
            contador_ids += 1
        
        linha = i // largura
        coluna = i % largura
        rotulos_map[linha, coluna] = mapa_ids_finais[raiz]
            
    return rotulos_map