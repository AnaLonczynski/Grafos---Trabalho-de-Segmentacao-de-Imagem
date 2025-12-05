"""
integracao.py
Objetivo: Testar a união do trabalho da Pessoa 1 com a Pessoa 2.
"""

import sys
import os
import numpy as np

# Imports dos modulos
try:
    import base_dados
except ImportError:
    print("ERRO: O arquivo 'base_dados.py' não foi encontrado neste diretório.")
    sys.exit(1)

try:
    import Edmonds
except ImportError:
    print("ERRO: O arquivo 'Edmonds' não foi encontrado.")
    sys.exit(1)

try:
    import segmentacao 
except ImportError:
    print("ERRO CRÍTICO: O arquivo 'segmentacao.py' não foi encontrado.")
    sys.exit(1)

try:
    import visualizacao
except ImportError:
    print("ERRO CRÍTICO: O arquivo 'visualizacao.py' não foi encontrado.")
    sys.exit(1)


# Método Main

def main():

    caminho_imagem = "totoro_rebaixado.jpg" 
    max_lado = None  # Tem que ser ajustado a depender da imagem
    vizinhanca = "8"
    
    print("=========================================")
    print(" INICIANDO INTEGRAÇÃO")
    print("=========================================")

    # ---------------------------------------------------------
    # 1. Executar Engenharia de Dados 
    # ---------------------------------------------------------
    print("\n>>> [1/3] Executando base_dados...")
    
    # Verifica se a imagem existe, se não, cria uma dummy para o código não quebrar
    if not os.path.exists(caminho_imagem):
        print(f"AVISO: Imagem '{caminho_imagem}' não encontrada.")
        print("Criando imagem aleatória temporária para teste de lógica...")
        img_temp = np.random.randint(0, 255, (max_lado, max_lado, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite("teste_temp.jpg", img_temp)
        caminho_imagem = "teste_temp.jpg"

    # Cria grafo e calcula pesos
    img, lista_pesos = base_dados.pipeline_unificado(
        caminho_imagem=caminho_imagem,
        caminho_saida_base="dados_teste",
        max_lado=max_lado,
        vizinhanca=vizinhanca,
        gerar_plots=False
    )
    
    h, w, _ = img.shape
    num_nos = h * w
    print(f"   -> Grafo gerado: {num_nos} nós (pixels).")
    print(f"   -> Total de arestas calculadas: {len(lista_pesos)}")

    # ---------------------------------------------------------
    # 2. Executar Algoritmo Core A 
    # ---------------------------------------------------------
    print("\n>>> [2/3] Executando ChiuLiu...")
    
    edmonds = Edmonds.EdmondsCore(num_nos=num_nos, raiz=0)
    
    edmonds.construir_grafo_entrada(lista_pesos)
    
    # Fase de Seleção
    pais = edmonds.selecionar_pais_minimos()
    print(f"   -> Seleção gulosa concluída. {len(pais)} arestas escolhidas.")
    
    # Fase de Detecção de Ciclo
    ciclo = edmonds.detectar_primeiro_ciclo(pais)

    # ---------------------------------------------------------
    # 3. Análise dos Resultados
    # ---------------------------------------------------------
    print("\n>>> [3/3] RELATÓRIO FINAL")
    print("-----------------------------------------")
    
    if ciclo:
        print(f"🔴 RESULTADO: Ciclo Detectado!")
        print(f"   Tamanho do ciclo: {len(ciclo)} nós")
        print(f"   Nós envolvidos (ID): {ciclo}")
        
        # Converter IDs para coordenadas (Linha, Coluna) para ficar legível
        coords_ciclo = [base_dados.id_para_coord(idx, w) for idx in ciclo]
        print(f"   Coords (L, C): {coords_ciclo}")
        
        print("   -> Contrair esses nós em um Super-Nó.")
        print("   -> Ajustar pesos das arestas que entram/saem desse grupo.")
        print("   -> Chamar recursão.")
    else:
        print(f"🟢 RESULTADO: Nenhum ciclo encontrado!")
        print("   A seleção gulosa formou uma Arborescência válida.")

    print("=========================================")

    # ---------------------------------------------------------
    # 4. Segmentação
    # ---------------------------------------------------------
    
    LIMIAR_K = 0.08

    print("\n>>> Executando Segmentação...")
    
    try:
        # Segmentar
        rotulos_map = segmentacao.segmentar_arborescencia(
            pais=pais, 
            limiar=LIMIAR_K, 
            num_pixels=num_nos, 
            dimensoes=(h, w)
        )
        
        num_segmentos = np.unique(rotulos_map).size
        print(f"   -> Segmentação concluída: {num_segmentos} segmentos criados.")

        # Visualizar
        print("\n   -> Gerando imagem final...")
        
        nome_saida = f"src/segmentacao/resultado_final_k{LIMIAR_K}.png"
        
        visualizacao.visualizar_segmentacao_lab(
            img_rgb_normalizada=img,   
            rotulos_map=rotulos_map,   
            salvar_arquivo=nome_saida  
        )

    except Exception as e:
        print(f"\n❌ ERRO durante a etapa de segmentação: {e}")
        traceback.print_exc()

    print("=========================================")
    print("\n=========================================")
    print(" INTEGRAÇÃO CONCLUÍDA")
    print("=========================================")

if __name__ == "__main__":
    main()