import numpy as np
from typing import List, Tuple

# Esta classe Union-Find (também conhecida como Disjoint Set Union) é
# essencial tanto para a Pessoa 4 (Kruskal) quanto para a Pessoa 5 (Segmentação).
# Ela gerencia eficientemente quais pixels pertencem ao mesmo conjunto/segmento.
class UnionFind:
    """
    Estrutura de dados Union-Find (Disjoint Set Union) com otimizações
    de "union by size" (união por tamanho) e "path compression" (compressão de caminho).

    Usada para agrupar pixels em segmentos.
    """
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
                root_i, root_j = root_j, root_i  # Garante que root_i é a maior árvore
            
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            self.num_components -= 1
            return True
        
        return False

# --- Implementação da Pessoa 5 ---

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

    # 5. & 6. Criar a matriz de rótulos (rotulos_map)
    
    # Inicializa a matriz de saída
    rotulos_map = np.zeros(dimensoes, dtype=int)
    
    # Mapeia as raízes do Union-Find (que podem ser números arbitrários)
    # para IDs de segmento sequenciais (0, 1, 2, ...)
    segmento_id_map = {}
    id_atual = 0
    
    for linha in range(altura):
        for coluna in range(largura):
            # Converte (linha, coluna) para o ID 1D do pixel
            pixel_id = linha * largura + coluna
            
            # Encontra a raiz do conjunto ao qual este pixel pertence
            raiz = uf.find(pixel_id)
            
            # Se esta é a primeira vez que vemos esta raiz,
            # atribui um novo ID de segmento sequencial a ela
            if raiz not in segmento_id_map:
                segmento_id_map[raiz] = id_atual
                id_atual += 1
            
            # Atribui o ID do segmento ao pixel na matriz de rótulos
            rotulos_map[linha, coluna] = segmento_id_map[raiz]
            
    return rotulos_map

# --- Bloco de Exemplo (Main) ---
if __name__ == "__main__":
    
    # 1. Simular as entradas da Pessoa 4 e do usuário
    
    # Vamos simular uma imagem pequena de 3x3 (9 pixels)
    # Pixels 0, 1, 2
    #        3, 4, 5
    #        6, 7, 8
    num_pixels = 9
    dimensoes = (3, 3)

    # Simulação da saída da MST (Pessoa 4)
    # (peso, u, v)
    # Suponha que os pixels [0,1,3,4] são uma região (ex: céu, peso baixo)
    # e [5,7,8] são outra (ex: grama, peso baixo)
    # e as arestas *entre* elas têm peso alto.
    mst_simulada = [
        # Arestas da região "céu" (pesos baixos)
        (0.05, 0, 1),
        (0.02, 0, 3),
        (0.04, 1, 4),
        (0.01, 3, 4),
        
        # Arestas da região "grama" (pesos baixos)
        (0.03, 7, 8),
        (0.06, 5, 8),

        # Arestas "cortadas" entre as regiões (pesos altos)
        (0.85, 1, 2), # 2 fica sozinho
        (0.90, 4, 5), # Conexão céu-grama
        (0.70, 3, 6)  # 6 fica sozinho
        
        # Nota: Uma MST real teria exatamente N-1 (ou 8) arestas.
        # Esta é uma simulação simplificada com 9 arestas.
        # Vamos ajustar para 8 arestas para ser uma MST válida.
    ]
    
    # MST válida com 8 arestas (N-1)
    mst_valida_simulada = [
        (0.01, 3, 4),
        (0.02, 0, 3),
        (0.03, 7, 8),
        (0.04, 1, 4),
        (0.95, 0, 1), # Esta aresta seria redundante se 0-3 e 1-4 já existem
        (0.06, 5, 8),
        # Vamos assumir que (0,1) não está na MST real, 
        # pois 0-3-4-1 já conecta 0 e 1.
        
        # MST mais realista (8 arestas)
        (0.90, 3, 4), # {3,4}
        (0.92, 0, 3), # {0,3,4}
        (0.03, 7, 8), # {7,8}, {0,3,4}
        (0.04, 1, 4), # {0,1,3,4}, {7,8}
        (0.06, 5, 8), # {0,1,3,4}, {5,7,8}
        
        # Arestas caras conectando os grupos e os pixels solitários
        (0.70, 3, 6), # {0,1,3,4}, {5,7,8}, {6}
        (0.85, 1, 2), # {0,1,3,4}, {5,7,8}, {6}, {2}
        (0.70, 4, 5)  # MST completa, conectando tudo
    ]
    
    
    # 2. Definir o limiar (Threshold)
    # Este valor é experimental. Vamos escolher um valor que
    # corte as arestas "caras" ( > 0.1)
    limiar_k = 0.1
    
    # 3. Executar a função da Pessoa 5
    rotulos_map = segmentar_mst(mst_valida_simulada, limiar_k, num_pixels, dimensoes)
    
    # 4. Exibir a saída (o que será entregue à Pessoa 6)
    print("\n--- Saída da Pessoa 5 (segmentacao.py) ---")
    print("Matriz 'rotulos_map':")
    print(rotulos_map)

    # Resultado esperado com limiar = 0.1:
    # Arestas usadas: (0.01, 3, 4), (0.02, 0, 3), (0.03, 7, 8), (0.04, 1, 4), (0.06, 5, 8)
    # Conjuntos finais:
    # {0, 1, 3, 4}  (Região 1)
    # {5, 7, 8}   (Região 2)
    # {2}           (Região 3)
    # {6}           (Região 4)
    #
    # Mapeamento esperado (os IDs podem variar, mas o agrupamento deve ser este):
    # Pixel 0 (Raiz R1) -> ID 0
    # Pixel 1 (Raiz R1) -> ID 0
    # Pixel 2 (Raiz R2) -> ID 1
    # Pixel 3 (Raiz R1) -> ID 0
    # Pixel 4 (Raiz R1) -> ID 0
    # Pixel 5 (Raiz R3) -> ID 2
    # Pixel 6 (Raiz R4) -> ID 3
    # Pixel 7 (Raiz R3) -> ID 2
    # Pixel 8 (Raiz R3) -> ID 2
    #
    # Matriz 'rotulos_map' esperada:
    # [[0, 0, 1],
    #  [0, 0, 2],
    #  [3, 2, 2]]