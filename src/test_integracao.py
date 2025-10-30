# Arquivo: test_integracao.py
#
# Este arquivo serve como o "programa principal" para testar
# a integração entre o trabalho da Pessoa 1, 2, 3, 4 e 5.
#
# Ele simula como o sistema final funcionará:
# 1. Pega a imagem (Pessoa 1)
# 2. Cria o grafo (Pessoa 2)
# 3. Calcula os pesos (Pessoa 3)
# 4. Encontra a MST (Pessoa 4)
# 5. Segmenta a MST (Pessoa 5)

import sys
import numpy as np
import os
from tqdm import tqdm 

# Adiciona o diretório atual ao path para garantir que os imports funcionem
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# ==========================================================
# ETAPA 0: catch de possiveis erros
# ==========================================================

try:
    # Pessoa 1
    from preprocs import preprocessar_imagem
except ImportError:
    print("ERRO: Não foi possível encontrar o arquivo 'preprocs.py'.")
    print("Verifique se o arquivo da Pessoa 1 está salvo como 'preprocs.py' na mesma pasta.")
    sys.exit(1)

try:
    # Pessoa 2
    from construir_grafo import criar_grafo_adjacencia
except ImportError:
    print("ERRO: Não foi possível encontrar o arquivo 'construir_grafo.py'.")
    print("Verifique se o seu arquivo (Pessoa 2) está salvo como 'construir_grafo.py' na mesma pasta.")
    sys.exit(1)

try:
    # Pessoa 3
    from pesos_grafo import calcular_pesos_arestas
except ImportError:
    print("ERRO: Não foi possível encontrar o arquivo 'pesos_grafo.py'.")
    print("Verifique se você criou este arquivo na mesma pasta.")
    sys.exit(1)

try:
    # Pessoa 4
    from mst_algoritmo import kruskal_mst
except ImportError:
    print("ERRO: Não foi possível encontrar o arquivo 'mst_algoritmo.py'.")
    print("Verifique se o arquivo da Pessoa 4 está na mesma pasta.")
    sys.exit(1)

try:
    # Pessoa 5
    from segmentacao import segmentar_mst
except ImportError:
    print("ERRO: Não foi possível encontrar o arquivo 'segmentacao.py'.")
    print("Verifique se o arquivo da Pessoa 5 está na mesma pasta.")
    sys.exit(1)

# --- INÍCIO DO TESTE ---

# Defina o nome da imagem a ser testada
NOME_ARQUIVO_TESTE = "minitotoro.jpeg" # Você pode trocar para "balls.png" ou outra
LIMIAR_K = 0.05 # Valor experimental para segmentação (Pessoa 5)

print("--- INICIANDO TESTE DE INTEGRAÇÃO (PESSOA 1 + 2 + 3 + 4 + 5) ---")

# ==========================================================
# ETAPA 1: Pré-processamento (Pessoa 1)
# ==========================================================

tqdm.write(f"\n[Pessoa 1] Processando a imagem: '{NOME_ARQUIVO_TESTE}'...")
matriz_processada = preprocessar_imagem(NOME_ARQUIVO_TESTE)

if matriz_processada is None:
    tqdm.write(f"ERRO FATAL: Falha ao carregar '{NOME_ARQUIVO_TESTE}'.")
    tqdm.write("Verifique se o nome do arquivo está correto e se ele existe na pasta.")
    sys.exit(1)

tqdm.write("[Pessoa 1] Imagem processada com sucesso.")


# ==========================================================
# ETAPA 2: Estrutura do Grafo (Pessoa 2)
# ==========================================================
tqdm.write("\n[Transição] Extraindo dimensões da matriz...")

altura, largura, _ = matriz_processada.shape
total_pixels = altura * largura
dimensoes = (altura, largura)

tqdm.write(f"Dimensões identificadas: Altura={altura}, Largura={largura}")
tqdm.write(f"Total de pixels (nós do grafo): {total_pixels}")

tqdm.write(f"\n[Pessoa 2] Criando o grafo de adjacência (8-vizinhos)...")
lista_arestas = criar_grafo_adjacencia(altura, largura)
tqdm.write("[Pessoa 2] Grafo estrutural criado com sucesso.")


# ==========================================================
# ETAPA 3: Cálculo de Pesos (Pessoa 3)
# ==========================================================

tqdm.write(f"\n[Pessoa 3] Calculando pesos (dist. Euclidiana) para {len(lista_arestas)} arestas...")
barra_pesos = tqdm(total=len(lista_arestas), desc="Calculando Pesos", unit=" arestas", leave=True, file=sys.stdout, ncols=80)
lista_arestas_com_pesos = calcular_pesos_arestas(matriz_processada, lista_arestas, barra_pesos)
barra_pesos.close()
tqdm.write("[Pessoa 3] Cálculo de pesos concluído.")


# ==========================================================
# ETAPA 4: Algoritmo de Kruskal (Pessoa 4)
# ==========================================================

tqdm.write("\n[Pessoa 4] Executando o algoritmo de Kruskal (Árvore Geradora Mínima)...")
barra_kruskal = tqdm(total=len(lista_arestas_com_pesos), desc="Executando Kruskal", unit=" arestas", leave=True, file=sys.stdout, ncols=80)
mst = kruskal_mst(lista_arestas_com_pesos, total_pixels, barra_kruskal)
barra_kruskal.close()
tqdm.write(f"[Pessoa 4] Kruskal executado com sucesso.")


# ==========================================================
# ETAPA 5: Segmentação (Pessoa 5)
# ==========================================================
tqdm.write("\n[Pessoa 5] Executando a segmentação baseada na MST...")
tqdm.write(f"Usando Limiar (k): {LIMIAR_K}")
tqdm.write(f"Total de arestas na MST para processar: {len(mst)}")

# Chama a sua função do arquivo 'segmentacao.py'
# Entradas: mst (P4), limiar (definido acima), total_pixels (P2), dimensoes (P2)
rotulos_map = segmentar_mst(mst, LIMIAR_K, total_pixels, dimensoes)

# Saída: rotulos_map (para Pessoa 6)
num_segmentos_final = np.unique(rotulos_map).size

tqdm.write(f"[Pessoa 5] Segmentação concluída.")
tqdm.write(f"[Pessoa 5] Mapa de rótulos (rotulos_map) criado com shape: {rotulos_map.shape}")


# ==========================================================
tqdm.write("\n--- TESTE DE INTEGRAÇÃO (1 + 2 + 3 + 4 + 5) FINALIZADO COM SUCESSO ---")
# ==========================================================


# ==========================================================
# ETAPA 6: Resultados Finais
# ==========================================================
tqdm.write("\n--- RESULTADOS DO TESTE ---")

tqdm.write(f"\n--- Etapas 1-4 (Criação da MST) ---")
tqdm.write(f"Imagem de Entrada: {NOME_ARQUIVO_TESTE}")
tqdm.write(f"Total de Nós (Pixels): {total_pixels}")
tqdm.write(f"Total de Arestas (Conexões): {len(lista_arestas)}")

if total_pixels > 0:
    ratio = len(lista_arestas) / total_pixels
    tqdm.write(f"Taxa (Arestas / Nós): {ratio:.4f}")
    tqdm.write("(Para 8-vizinhos, este valor deve ser próximo de 4.0 para imagens grandes)")
    tqdm.write(f"Tamanho da MST: {len(mst)} arestas (Nós - 1 = {total_pixels - 1})")
    
    peso_total_mst = sum([peso for peso, _, _ in mst])
    tqdm.write(f"Peso total da MST: {peso_total_mst:.6f}")

tqdm.write(f"\n--- Etapa 5 (Segmentação) ---")
tqdm.write(f"Limiar (k) utilizado: {LIMIAR_K}")
tqdm.write(f"Shape do Mapa de Rótulos: {rotulos_map.shape}")
tqdm.write(f"Total de Segmentos Finais: {num_segmentos_final}")

# Imprime um pedaço da matriz de rótulos
tqdm.write("\n--- Amostra do Mapa de Rótulos (Pessoa 5) ---")
tqdm.write("Exibindo o canto superior esquerdo (até 15x15 pixels):")
tqdm.write("(Cada número representa um ID de segmento diferente)")

# Pega um slice de até 15x15 da matriz
max_linhas_amostra = min(altura, 16)
max_colunas_amostra = min(largura, 16)
amostra_rotulos = rotulos_map[0:max_linhas_amostra, 0:max_colunas_amostra]

# Configura o NumPy para imprimir a matriz de forma mais limpa
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
print(amostra_rotulos)
# Restaura as opções padrão do NumPy (opcional, mas boa prática)
np.set_printoptions(threshold=1000, linewidth=75)