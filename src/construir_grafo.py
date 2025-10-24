
# ------------------------------------------------------------------------------
#| Aqui se recebe a largura e a altura gerada após o processamento da Imagem.   |
#| Após isso, cria-se a estrutura de adjacência (as arestas) de um grafo baseado|
#| em uma grade (imagem), conectando pixels vizinhos.                           |
# ------------------------------------------------------------------------------

def criar_grafo_adjacencia(altura, largura):
    """
    Cria a estrutura de adjacência (as arestas) de um grafo
    baseado em uma grade (imagem), conectando pixels vizinhos.

    Para evitar arestas duplicadas (ex: (A,B) e (B,A)), cada pixel
    só se conecta a 4 dos seus 8 vizinhos:
    1. Direita
    2. Baixo-Esquerda
    3. Baixo
    4. Baixo-Direita

    @Return:
    - list: Uma lista de tuplas, onde cada tupla (u, v) representa
            uma aresta entre o pixel com ID 'u' e o pixel com ID 'v'.
    """
    print(f"Criando grafo para uma imagem de {altura}x{largura}...")

    arestas = []

    for linha in range(altura):
        for coluna in range(largura):

            # Id unico do pixel atual: 
            # Esta fórumila nos garante que todo  pixel terá um ID diferente
            id_atual = (linha * largura) + coluna

            # 1. Vizinho da DIREITA (Horizontal)
            # Checa se não está na última coluna
            if coluna + 1 < largura:
                id_vizinho_R = id_atual + 1
                arestas.append((id_atual, id_vizinho_R))

            # 2. Vizinho de BAIXO-ESQUERDA (Diagonal)
            # Checa se não está na última linha E não está na primeira coluna
            if (linha + 1 < altura) and (coluna - 1 >= 0):
                id_vizinho_BL = (linha + 1) * largura + (coluna - 1)
                arestas.append((id_atual, id_vizinho_BL))

            # 3. Vizinho de BAIXO (Vertical)
            # Checa se não está na última linha
            if linha + 1 < altura:
                id_vizinho_B = id_atual + largura
                arestas.append((id_atual, id_vizinho_B))

            # 4. Vizinho de BAIXO-DIREITA (Diagonal)
            # Checa se não está na última linha E não está na última coluna
            if (linha + 1 < altura) and (coluna + 1 < largura):
                id_vizinho_BR = (linha + 1) * largura + (coluna + 1)
                arestas.append((id_atual, id_vizinho_BR))

    print(f"Grafo estrutural 8-vizinhos criado com {len(arestas)} arestas.")
    
    return arestas