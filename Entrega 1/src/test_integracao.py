
# Este arquivo serve como o "programa principal" para testar
# a integração entre o trabalho 

import sys
import numpy as np
import os
import cv2 
from tqdm import tqdm 

# Adiciona o diretório atual ao path para garantir que os imports funcionem
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# ==========================================================
# ETAPA 0: catch de possiveis erros
# ==========================================================
tqdm.write("Carregando módulos...")
try:
    from preprocs import preprocessar_imagem
except ImportError:
    tqdm.write("ERRO: Não foi possível encontrar o arquivo 'preprocs.py'.")
    sys.exit(1)
try:
    # Pessoa 2
    from construir_grafo import criar_grafo_adjacencia
except ImportError:
    tqdm.write("ERRO: Não foi possível encontrar o arquivo 'construir_grafo.py'.")
    sys.exit(1)
try:
    # Pessoa 3
    from pesos_grafo import calcular_pesos_arestas
except ImportError:
    tqdm.write("ERRO: Não foi possível encontrar o arquivo 'pesos_grafo.py'.")
    sys.exit(1)
try:
    # Pessoa 4
    from mst_algoritmo import kruskal_mst
except ImportError:
    tqdm.write("ERRO: Não foi possível encontrar o arquivo 'mst_algoritmo.py'.")
    sys.exit(1)
try:
    # Pessoa 5
    from segmentacao import segmentar_mst
except ImportError:
    tqdm.write("ERRO: Não foi possível encontrar o arquivo 'segmentacao.py'.")
    sys.exit(1)
try:
    # Pessoa 6 
    from visualizacao import visualizar_segmentacao_lab
except ImportError:
    tqdm.write("ERRO: Não foi possível encontrar o arquivo 'visualizacao.py'.")
    sys.exit(1)

# --- INÍCIO DO TESTE ---

NOME_ARQUIVO_TESTE = "totoro_rebaixado.jpg" 
LIMIAR_K = 0.015 

print(f"--- INICIANDO TESTE DE INTEGRAÇÃO ---")
tqdm.write(f"Imagem: {NOME_ARQUIVO_TESTE} | Limiar K: {LIMIAR_K}")

# ==========================================================
# ETAPA 1: Pré-processamento 
# ==========================================================

tqdm.write(f"\nProcessando a imagem: '{NOME_ARQUIVO_TESTE}'...")

matriz_processada_lab = preprocessar_imagem(NOME_ARQUIVO_TESTE)

if matriz_processada_lab is None:
    tqdm.write(f"ERRO FATAL: Falha ao carregar '{NOME_ARQUIVO_TESTE}'.")
    sys.exit(1)

tqdm.write(" Matriz RBG processada criada com sucesso.")

# ==========================================================
# ETAPA 1b: Preparação da Imagem 
# ==========================================================

tqdm.write(f"\n[Teste] Carregando imagem RGB original ...")
img_bgr_p6 = cv2.imread(NOME_ARQUIVO_TESTE)
if img_bgr_p6 is None:
    tqdm.write(f"ERRO FATAL: Falha ao carregar '{NOME_ARQUIVO_TESTE}' pela segunda vez.")
    sys.exit(1)

img_rgb_p6 = cv2.cvtColor(img_bgr_p6, cv2.COLOR_BGR2RGB)
# Normalizar para [0, 1] 
matriz_rgb_normalizada = img_rgb_p6.astype(np.float32) / 255.0
tqdm.write("[Teste] Matriz RGB original criada com sucesso.")

# ==========================================================
# ETAPA 2: Estrutura do Grafo
# ==========================================================
tqdm.write("\n[Transição] Extraindo dimensões da matriz...")

altura, largura, _ = matriz_processada_lab.shape
total_pixels = altura * largura
dimensoes = (altura, largura)

tqdm.write(f"Dimensões identificadas: Altura={altura}, Largura={largura}")
tqdm.write(f"Total de pixels (nós do grafo): {total_pixels}")

tqdm.write(f"\n Criando o grafo de adjacência (8-vizinhos)...")
lista_arestas = criar_grafo_adjacencia(altura, largura)
tqdm.write(" Grafo estrutural criado com sucesso.")


# ==========================================================
# ETAPA 3: Cálculo de Pesos 
# ==========================================================

tqdm.write(f"\nCalculando pesos (Dist. RGB) para {len(lista_arestas)} arestas...")
barra_pesos = tqdm(total=len(lista_arestas), desc="Calculando Pesos", unit=" arestas", leave=True, file=sys.stdout, ncols=80)

lista_arestas_com_pesos = calcular_pesos_arestas(matriz_processada_lab, lista_arestas, barra_pesos)
barra_pesos.close()
tqdm.write("Cálculo de pesos concluído.")


# ==========================================================
# ETAPA 4: Algoritmo de Kruskal 
# ==========================================================

tqdm.write("\nExecutando o algoritmo de Kruskal (Árvore Geradora Mínima)...")
barra_kruskal = tqdm(total=len(lista_arestas_com_pesos), desc="Executando Kruskal", unit=" arestas", leave=True, file=sys.stdout, ncols=80)
mst = kruskal_mst(lista_arestas_com_pesos, total_pixels, barra_kruskal)
barra_kruskal.close()
tqdm.write(f"[Kruskal executado com sucesso.")


# ==========================================================
# ETAPA 5: Segmentação
# ==========================================================
tqdm.write("\nExecutando a segmentação baseada na MST...")
tqdm.write(f"Usando Limiar (k): {LIMIAR_K}")
tqdm.write(f"Total de arestas na MST para processar: {len(mst)}")

rotulos_map = segmentar_mst(mst, LIMIAR_K, total_pixels, dimensoes)
num_segmentos_final = np.unique(rotulos_map).size
tqdm.write(f"Segmentação concluída. {num_segmentos_final} segmentos encontrados.")


# ==========================================================
# ETAPA 6: Visualização 
# ==========================================================
tqdm.write("\nIniciando a etapa de visualização final (média L*a*b*)...")

nome_base = os.path.splitext(NOME_ARQUIVO_TESTE)[0]
ARQUIVO_SAIDA = f"resultado_{nome_base}_k{LIMIAR_K}_lab.png"

try:
    # Passamos a matriz RGB  (matriz_rgb_normalizada)
    # e o mapa de rótulos (rotulos_map)
    visualizar_segmentacao_lab(matriz_rgb_normalizada, 
                               rotulos_map, 
                               salvar_arquivo=ARQUIVO_SAIDA)
except Exception as e:
     tqdm.write(f"\nERRO INESPERADO durante a Etapa 6 (Visualização): {e}")
     tqdm.write("Verifique se as bibliotecas (matplotlib, scikit-image, scipy) estão instaladas.")
     sys.exit(1)
     
tqdm.write("\n[Processo Concluído]")