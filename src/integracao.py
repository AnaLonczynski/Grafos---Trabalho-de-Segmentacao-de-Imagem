"""
integracao.py
Objetivo: Testar a união do trabalho da Pessoa 1 com a Pessoa 2.
"""

import sys
import os
import numpy as np

# Importa o módulo da Pessoa 1
try:
    import base_dados
except ImportError:
    print("ERRO: O arquivo 'base_dados.py' não foi encontrado neste diretório.")
    sys.exit(1)

# Importa o módulo da Pessoa 2
try:
    import Edmonds
except ImportError:
    print("ERRO: O arquivo 'Edmonds' não foi encontrado.")
    sys.exit(1)

def main():
    # --- PARÂMETROS ---
    # Ajuste o caminho da imagem aqui para testar
    caminho_imagem = "totoro_rebaixado.jpg" 
    
    # IMPORTANTE: Para testes rápidos e para conseguir VISUALIZAR ciclos no terminal,
    # use um tamanho pequeno (ex: 50 ou 100 pixels de lado).
    max_lado = 50 
    vizinhanca = "4"
    
    print("=========================================")
    print(" INICIANDO INTEGRAÇÃO PESSOA 1 + PESSOA 2")
    print("=========================================")

    # ---------------------------------------------------------
    # 1. Executar Engenharia de Dados (Pessoa 1)
    # ---------------------------------------------------------
    print("\n>>> [1/3] Exectando Pessoa 1 (base_dados)...")
    
    # Verifica se a imagem existe, se não, cria uma dummy para o código não quebrar
    if not os.path.exists(caminho_imagem):
        print(f"AVISO: Imagem '{caminho_imagem}' não encontrada.")
        print("Criando imagem aleatória temporária para teste de lógica...")
        img_temp = np.random.randint(0, 255, (max_lado, max_lado, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite("teste_temp.jpg", img_temp)
        caminho_imagem = "teste_temp.jpg"

    # Chama a função pipeline do arquivo da Pessoa 1
    # Nota: O pipeline_unificado já carrega, cria grafo e calcula pesos
    img, lista_pesos = base_dados.pipeline_unificado(
        caminho_imagem=caminho_imagem,
        caminho_saida_base="dados_teste",
        max_lado=max_lado,
        vizinhanca=vizinhanca,
        gerar_plots=False # Desliga plots da P1 para focar no terminal
    )
    
    h, w, _ = img.shape
    num_nos = h * w
    print(f"   -> Grafo gerado: {num_nos} nós (pixels).")
    print(f"   -> Total de arestas calculadas: {len(lista_pesos)}")

    # ---------------------------------------------------------
    # 2. Executar Algoritmo Core - Parte 2  
    # ---------------------------------------------------------
    print("\n>>> [2/3] Executando Pessoa 2 e 3 (Edmonds Recursivo)...")
    
    edmonds = Edmonds.EdmondsCore(num_nos=num_nos, raiz=0)
    edmonds.construir_grafo_entrada(lista_pesos)
    
    arborescencia_final = edmonds.resolver_arborescencia()

    # ---------------------------------------------------------
    # 3. Análise dos Resultados
    # ---------------------------------------------------------
    print("\n>>> [3/3] RELATÓRIO FINAL")
    print("-----------------------------------------")
    
    if arborescencia_final:
        print(f"🟢 SUCESSO: Arborescência Mínima construída!")
        print(f"   Total de arestas na solução: {len(arborescencia_final)}")
        
        custo_total = sum(item[1] for item in arborescencia_final.values())
        print(f"   Custo total: {custo_total:.4f}")
        
        print("\n   PRÓXIMO PASSO (Visualização):")
        print("   -> A Pessoa 4 deve pegar esse dicionário e desenhar a segmentação.")
    else:
        print("🔴 ERRO: Não foi possível gerar a arborescência (retorno vazio).")

    print("=========================================")

if __name__ == "__main__":
    main()