# Arquivo: mst_algoritmo.py
# -------------------------------------------------------------------------
# Pessoa 4 — Algoritmo de Kruskal (Árvore Geradora Mínima - MST)
# -------------------------------------------------------------------------
# A lista de arestas DEVE ser passada JÁ ordenada por peso.
# -------------------------------------------------------------------------

class UnionFind:
    """Estrutura Union-Find (Set Union Disjunta) para controle de componentes."""
    def __init__(self, n):
        # Cada nó começa como seu próprio pai (conjunto individual)
        self.parent = list(range(n))
        # O rank é usado para otimizar as uniões
        self.rank = [0] * n

    def find(self, x):
        """Encontra o representante (raiz) do conjunto de x com compressão de caminho."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # compressão de caminho
        return self.parent[x]

    def union(self, x, y):
        """Une os conjuntos de x e y. Retorna True se a união ocorreu."""
        raiz_x = self.find(x)
        raiz_y = self.find(y)

        if raiz_x == raiz_y:
            return False  # já estão no mesmo conjunto → criaria ciclo

        # União por rank (mantém árvore balanceada)
        if self.rank[raiz_x] < self.rank[raiz_y]:
            self.parent[raiz_x] = raiz_y
        elif self.rank[raiz_x] > self.rank[raiz_y]:
            self.parent[raiz_y] = raiz_x
        else:
            self.parent[raiz_y] = raiz_x
            self.rank[raiz_x] += 1

        return True


def kruskal_mst(arestas_ponderadas, num_nos):
    """
    Executa o algoritmo de Kruskal para encontrar a Árvore Geradora Mínima (MST).

    Parâmetros:
    - arestas_ponderadas (list): lista de tuplas (peso, u, v) JÁ ORDENADA.
    - num_nos (int): número total de pixels.
    - barra_progresso (tqdm object, optional): Objeto tqdm para atualizar a barra de progresso.

    Retorna:
    - mst (list): lista de arestas (peso, u, v) que formam a MST.
    """
    # 1. A ordenação deve ser feita no arquivo principal.
    arestas_ordenadas = arestas_ponderadas

    # 2. Inicializar a estrutura Union-Find
    uf = UnionFind(num_nos)

    # 3. Lista para armazenar a MST
    mst = []

    # 4. Percorrer as arestas em ordem crescente de peso
    for peso, u, v in arestas_ordenadas:
        
        # Se u e v pertencem a conjuntos diferentes → não forma ciclo
        if uf.union(u, v):
            mst.append((peso, u, v))
            
            # Critério de parada: uma MST tem (num_nos - 1) arestas
            if len(mst) == num_nos - 1:
                # O loop irá parar aqui. A barra será fechada no código principal.
                break

        # ATUALIZAÇÃO: Atualiza a barra a cada iteração de aresta

    return mst