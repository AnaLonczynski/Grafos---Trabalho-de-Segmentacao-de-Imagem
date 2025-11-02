
# ------------------------------------------------------------------------------
#| O arquivo segmentar_mst.py realiza a segmentação da imagem a partir da       |
#| Árvore Geradora Mínima (MST), agrupando pixels conectados por arestas de     |
#| baixo peso de acordo com o limiar K                                          |
# ------------------------------------------------------------------------------
 
import numpy as np
from typing import List, Tuple

class UnionFind:
    
    #Usada para agrupar pixels em segmentos.
    
    def __init__(self, n: int):
        """
        Inicializa a estrutura para 'n' elementos (pixels).
        Cada pixel começa em seu próprio conjunto.
        """
        # 'parent[i]' armazena o pai do elemento 'i'.
        # Se parent[i] == i, então 'i' é a raiz de um conjunto.
        self.parent = list(range(n))
        
        # 'size[i]' armazena o tamanho do conjunto cuja raiz é 'i'.
        # Usado para a otimização "union by size".
        self.size = [1] * n
        
        # O número de conjuntos distintos (segmentos).
        self.num_components = n

    def find(self, i: int) -> int:
        """
        Encontra a raiz (representante) do conjunto ao qual 'i' pertence.
        Aplica "path compression" no processo.
        """
        if self.parent[i] == i:
            return i
        
        # Path Compression: Faz com que 'i' e seus ancestrais
        # apontem diretamente para a raiz.
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        """
        Une os conjuntos que contêm 'i' e 'j'.
        Usa "union by size": o conjunto menor é anexado ao maior.
        Retorna True se uma união foi realizada, False se 'i' e 'j'
        já estavam no mesmo conjunto.
        """
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            # Union by Size: Anexa a árvore menor à raiz da árvore maior.
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i 
            
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            self.num_components -= 1
            return True
        
        return False

def segmentar_mst(mst: List[Tuple[float, int, int]], 
                  limiar: float, 
                  num_pixels: int, 
                  dimensoes: Tuple[int, int]) -> np.ndarray:
    """
    Transforma a Árvore Geradora Mínima (MST) em uma segmentação.

    Itera pela MST e une pixels (nós) em conjuntos se a aresta (peso)
    entre eles for menor ou igual ao 'limiar'. Arestas com peso maior
    são "cortadas", separando os segmentos.

    Args:
        mst: A lista de arestas da MST (da Pessoa 4), 
             formato [(peso, u, v), ...].
        limiar: O valor máximo de peso de aresta para considerar
                dois pixels como parte do mesmo segmento.
        num_pixels: O número total de pixels (altura * largura).
        dimensoes: Uma tupla (altura, largura) da imagem.

    Returns:
        Uma matriz 2D (numpy array) 'rotulos_map', onde cada posição
        contém o ID do segmento ao qual aquele pixel pertence.
    """
    
    print(f"Iniciando segmentação com limiar = {limiar}...")
    
    altura, largura = dimensoes
    
    # 1. Criar uma nova estrutura Union-Find
    uf = UnionFind(num_pixels)
    
    # 2. & 3. Percorrer a MST e unir conjuntos se o peso <= limiar
    arestas_unidas = 0
    for peso, u, v in mst:
        # 4. Se o peso for baixo o suficiente, une os pixels no mesmo segmento
        if peso <= limiar:
            uf.union(u, v)
            arestas_unidas += 1

    print(f"Segmentação concluída. {arestas_unidas} arestas da MST unidas.")
    print(f"Número total de segmentos encontrados: {uf.num_components}")

    rotulos_map = np.zeros(dimensoes, dtype=int)
    
    segmento_id_map = {}
    id_atual = 0
    
    for linha in range(altura):
        for coluna in range(largura):

            pixel_id = linha * largura + coluna
            
            raiz = uf.find(pixel_id)
            
            if raiz not in segmento_id_map:
                segmento_id_map[raiz] = id_atual
                id_atual += 1
            
            rotulos_map[linha, coluna] = segmento_id_map[raiz]
            
    return rotulos_map