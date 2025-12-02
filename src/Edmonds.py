"""
Algoritmo de Edmond e Chiliu Chuliu Chiuliu
Descrição: Implementa a fase inicial do algoritmo de Chu-Liu / Edmonds.
           1. Seleção Gulosa (Greedy) dos pais de menor custo.
           2. Detecção de Ciclos na seleção feita.

           RAYSSA E BRUNO, PODEM CONTINUAR NESSE AQUI MSM AAAA
"""

from typing import List, Tuple, Dict, Optional

class EdmondsCore:
    def __init__(self, num_nos: int, raiz: int = 0):
        self.num_nos = num_nos
        self.raiz = raiz
        # arestas_entrada[v] = lista de tuplas (u, peso)
        # Significa que existe uma aresta u -> v com custo peso
        self.arestas_entrada: List[List[Tuple[int, float]]] = [[] for _ in range(num_nos)]

    def construir_grafo_entrada(self, lista_arestas_com_peso: List[Tuple[int, int, float]]):
        """
        Recebe a lista bruta da Pessoa 1 (u, v, w) e converte para
        lista de adjacência invertida para acesso rápido.
        """
        print(f"[ChiuLiu] Organizando grafo com {len(lista_arestas_com_peso)} arestas...")
        for u, v, w in lista_arestas_com_peso:
            self.arestas_entrada[v].append((u, w))

    def selecionar_pais_minimos(self) -> Dict[int, Tuple[int, float]]:
        """
        Passo 1: Para cada nó (exceto raiz), escolhe a aresta de entrada mais barata.
        Retorna: Dicionário {filho: (pai, peso)}
        """
        pais_escolhidos = {}
        
        # Itera sobre todos os nós do grafo
        for v in range(self.num_nos):
            if v == self.raiz:
                continue
            
            entradas = self.arestas_entrada[v]
            if not entradas:
                continue # Nó isolado (não deveria acontecer na segmentação de img padrão)
            
            # Acha a tupla com o menor peso 'w'
            # x[1] é o peso na tupla (u, w)
            melhor_pai, menor_peso = min(entradas, key=lambda x: x[1])
            
            pais_escolhidos[v] = (melhor_pai, menor_peso)
            
        return pais_escolhidos

    def detectar_primeiro_ciclo(self, pais: Dict[int, Tuple[int, float]]) -> Optional[List[int]]:
        """
        Passo 2: Verifica se a escolha gulosa criou loops.
        Retorna: Lista de IDs dos nós que compõem o ciclo (se houver), ou None.
        """
        visitados_geral = [False] * self.num_nos
        
        print("[ChiuLiu] Buscando ciclos na seleção...")

        for i in range(self.num_nos):
            if visitados_geral[i]:
                continue
            
            # Rastreamento do caminho atual para achar back-edges
            caminho_atual = []
            conjunto_caminho = set() # Para busca O(1)
            
            curr = i
            
            # Navega "para cima" seguindo os pais
            while curr is not None:
                if curr in conjunto_caminho:
                    # ACHAMOS UM CICLO!
                    # O ciclo começa na primeira ocorrência de 'curr' no caminho_atual até o fim
                    indice_inicio = caminho_atual.index(curr)
                    ciclo = caminho_atual[indice_inicio:]
                    return ciclo
                
                if visitados_geral[curr]:
                    # Encontramos um nó já processado anteriormente que não formou ciclo
                    # ou leva à raiz. Podemos parar este ramo.
                    break
                
                # Marcação
                visitados_geral[curr] = True
                caminho_atual.append(curr)
                conjunto_caminho.add(curr)
                
                # Sobe para o pai
                if curr in pais:
                    curr = pais[curr][0] # Pega o ID do pai
                else:
                    curr = None # Chegou na raiz ou nó sem pai
                    
        return None