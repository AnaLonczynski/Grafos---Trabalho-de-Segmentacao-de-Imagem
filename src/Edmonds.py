"""
Algoritmo de Edmond e Chu-Liu
Descrição: Implementação completa (Seleção, Detecção, Contração e Expansão).
"""

from typing import List, Tuple, Dict, Optional, Set

class EdmondsCore:
    def __init__(self, num_nos: int, raiz: int = 0):
        self.num_nos = num_nos
        self.raiz = raiz
        self.arestas_entrada: Dict[int, List[tuple]] = {i: [] for i in range(num_nos)}

    def construir_grafo_entrada(self, lista_arestas_com_peso: List[Tuple[int, int, float]]):
        self.arestas_entrada = {i: [] for i in range(self.num_nos)}
        for u, v, w in lista_arestas_com_peso:
            self.arestas_entrada[v].append((u, w, u, v))

    def _selecionar_pais_minimos(self, vertices_ativos: List[int]) -> Dict[int, Tuple[int, float, int, int]]:
        """
        Passo 1: Guloso. Escolhe o pai mais barato para cada nó ativo.
        """
        pais_escolhidos = {}
        for v in vertices_ativos:
            if v == self.raiz:
                continue
            
            entradas = self.arestas_entrada.get(v, [])
            if not entradas:
                continue 
            
            # Pega a aresta de menor peso
            melhor_aresta = min(entradas, key=lambda x: x[1])
            # melhor_aresta = (u, w, orig_u, orig_v)
            pais_escolhidos[v] = melhor_aresta
            
        return pais_escolhidos

    def _detectar_ciclo(self, pais: Dict[int, tuple], vertices_ativos: List[int]) -> Optional[List[int]]:
        """
        Passo 2: Detecta se há ciclos na seleção gulosa.
        """
        visitados = {v: False for v in vertices_ativos}
        path_stack = []
        path_set = set()

        for i in vertices_ativos:
            if visitados[i]:
                continue
            
            curr = i
            path_stack = []
            path_set = set()
            
            while curr is not None:
                if curr in path_set:
                    # Ciclo encontrado!
                    indice_inicio = path_stack.index(curr)
                    return path_stack[indice_inicio:]
                
                if curr in visitados and visitados[curr]:
                    break
                
                visitados[curr] = True
                path_stack.append(curr)
                path_set.add(curr)
                
                # Avança para o pai (se existir)
                if curr in pais:
                    curr = pais[curr][0] # O pai é o primeiro elemento da tupla
                else:
                    curr = None
        
        return None

    def resolver_arborescencia(self, vertices_ativos: Optional[List[int]] = None) -> Dict[int, Tuple[int, float]]:
        """
        Lógica Recursiva Principal (A 'Mágica').
        """
        if vertices_ativos is None:
            vertices_ativos = list(range(self.num_nos))

        # 1. Seleção Gulosa
        pais = self._selecionar_pais_minimos(vertices_ativos)

        # 2. Verifica Ciclo
        ciclo = self._detectar_ciclo(pais, vertices_ativos)

        # CASO BASE: Se não tem ciclo, achamos a solução ótima para este nível
        if not ciclo:
            # Retorna formato simplificado {filho: (pai, peso)}
            return {k: (v[0], v[1]) for k, v in pais.items()}

        # CASO RECURSIVO: Tem ciclo. Precisamos contrair.
        print(f"   -> Ciclo detectado com {len(ciclo)} nós. Contraindo...")

        # ---------------------------------------------------------
        # FASE 3: CONTRAÇÃO (Criar Super-Nó)
        # ---------------------------------------------------------
        
        id_super_no = max(vertices_ativos) + 1
        nodes_ciclo_set = set(ciclo)
        
        # Novos vértices ativos = (Ativos - Ciclo) + {SuperNó}
        novos_vertices_ativos = [v for v in vertices_ativos if v not in nodes_ciclo_set]
        novos_vertices_ativos.append(id_super_no)

        novas_arestas = {v: [] for v in novos_vertices_ativos}
        
        referencia_arestas = {} 

        # Arestas que compõem o ciclo (para calcular o custo de redução)
        peso_no_ciclo = {} # Peso da aresta que aponta PARA v dentro do ciclo
        for v in ciclo:
            pai_v, peso_v, _, _ = pais[v]
            peso_no_ciclo[v] = peso_v

        # Reconstruir arestas para o grafo contraído
        for v in vertices_ativos:
            lista_entrada = self.arestas_entrada.get(v, [])
            for u, w, orig_u, orig_v in lista_entrada:
                if u not in vertices_ativos: continue # Ignora nós mortos

                u_novo = id_super_no if u in nodes_ciclo_set else u
                v_novo = id_super_no if v in nodes_ciclo_set else v

                if u_novo != v_novo:
                    novo_peso = w
                    if v_novo == id_super_no:
                        novo_peso = w - peso_no_ciclo[v]
                    novas_arestas[v_novo].append((u_novo, novo_peso, orig_u, orig_v))

        backup_arestas = self.arestas_entrada
        self.arestas_entrada = novas_arestas
        
        # ---------------------------------------------------------
        # CHAMADA RECURSIVA
        # ---------------------------------------------------------
        arborescencia_contraida = self.resolver_arborescencia(novos_vertices_ativos)
        
        # Restaura arestas originais (Backtracking)
        self.arestas_entrada = backup_arestas

        # ---------------------------------------------------------
        # FASE 4: EXPANSÃO (Descompactar Super-Nó)
        # ---------------------------------------------------------
        solucao_final = {}
        
        # Nó de entrada no ciclo (quem quebrou o ciclo)
        no_entrada_ciclo = None
        aresta_quebra_ciclo = None

        for filho_novo, (pai_novo, _) in arborescencia_contraida.items():
            if filho_novo == id_super_no:
                candidatos = []
                for v_ciclo in ciclo:
                    for u, w, ou, ov in self.arestas_entrada[v_ciclo]:
                        u_map = id_super_no if u in nodes_ciclo_set else u
                        if u_map == pai_novo:
                            peso_ajustado = w - peso_no_ciclo[v_ciclo]
                            candidatos.append( (peso_ajustado, u, v_ciclo, w) )
                
                _, pai_real, filho_real, peso_real = min(candidatos, key=lambda x: x[0])
                
                solucao_final[filho_real] = (pai_real, peso_real)
                no_entrada_ciclo = filho_real
            
            elif pai_novo == id_super_no:
                cand_saida = []
                for u, w, ou, ov in self.arestas_entrada[filho_novo]:
                    if u in nodes_ciclo_set:
                        cand_saida.append((u, w))
                
                melhor_saida, p_saida = min(cand_saida, key=lambda x: x[1])
                solucao_final[filho_novo] = (melhor_saida, p_saida)
                
            else:
                solucao_final[filho_novo] = arborescencia_contraida[filho_novo]

        for v in ciclo:
            if v != no_entrada_ciclo:
                pai_ciclo, peso_ciclo, _, _ = pais[v]
                solucao_final[v] = (pai_ciclo, peso_ciclo)

        return solucao_final