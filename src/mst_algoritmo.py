# -------------------------------------------------------------------------
# Pessoa 4 — Algoritmo de Kruskal (Árvore Geradora Mínima - MST)
# -------------------------------------------------------------------------
# Objetivo: Dado o grafo ponderado (arestas_ponderadas),
# encontrar a MST que conecta todos os pixels com o menor custo total.
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


def kruskal_mst(arestas_ponderadas, num_nos, barra_progresso=None):
    # Ordena as arestas pelo peso
    arestas_ordenadas = sorted(arestas_ponderadas, key=lambda x: x[0])

    #Inicializa a estrutura Union-Find
    uf = UnionFind(num_nos)

    #Lista para armazenar a MST
    mst = []

    #Percorrer as arestas em ordem crescente de peso
    for peso, u, v in arestas_ordenadas:
        # Se u e v pertencem a conjuntos diferentes → não forma ciclo
        if uf.union(u, v):
            mst.append((peso, u, v))
            # Critério de parada: uma MST tem (num_nos - 1) arestas
            if len(mst) == num_nos - 1:
                break

        if barra_progresso:
            barra_progresso.update(1)

    return mst
