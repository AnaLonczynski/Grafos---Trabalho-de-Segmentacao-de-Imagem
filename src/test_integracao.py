# Arquivo: test_integracao.py
#
# Este arquivo serve como o "programa principal" para testar
# a integração entre o trabalho da Pessoa 1 e da Pessoa 2.
#
# Ele simula como o sistema final funcionará:
# 1. Pega a imagem (Pessoa 1)
# 2. Pega as dimensões (Pessoa 1 -> Pessoa 2)
# 3. Cria o grafo (Pessoa 2)

import sys
import numpy as np
import os
from tqdm import tqdm #

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# ==========================================================
# ETAPA 0: catch de possiveis erros
# ==========================================================

try:
    from preprocs import preprocessar_imagem
except ImportError:
    print("ERRO: Não foi possível encontrar o arquivo 'preprocs.py'.")
    print("Verifique se o arquivo da Pessoa 1 está salvo como 'preprocs.py' na mesma pasta.")
    sys.exit(1)

try:
    from construir_grafo import criar_grafo_adjacencia
except ImportError:
    print("ERRO: Não foi possível encontrar o arquivo 'grafo_basico.py'.")
    print("Verifique se o seu arquivo (Pessoa 2) está salvo como 'grafo_basico.py' na mesma pasta.")
    sys.exit(1)

try:
    from pesos_grafo import calcular_pesos_arestas
except ImportError:
    print("ERRO: Não foi possível encontrar o arquivo 'grafo_pesos.py'.")
    print("Verifique se você criou este arquivo na mesma pasta.")
    sys.exit(1)

try:
    from mst_algoritmo import kruskal_mst
except ImportError:
    print("ERRO: Não foi possível encontrar o arquivo 'mst_algoritmo.py'.")
    print("Verifique se o arquivo da Pessoa 4 está na mesma pasta.")
    sys.exit(1)

# --- INÍCIO DO TESTE ---

# Defina o nome da imagem a ser testada
NOME_ARQUIVO_TESTE = "totoro_rebaixado.jpg"

print("--- INICIANDO TESTE DE INTEGRAÇÃO (PESSOA 1 + 2 + 3 + 4) ---")

# ==========================================================
# ETAPA 1: Pré-processamento
# ==========================================================

tqdm.write(f"\n[Pessoa 1] Processando a imagem: '{NOME_ARQUIVO_TESTE}'...")

matriz_processada = preprocessar_imagem(NOME_ARQUIVO_TESTE)

if matriz_processada is None:
    tqdm.write(f"ERRO FATAL: Falha ao carregar '{NOME_ARQUIVO_TESTE}'.")
    tqdm.write("Verifique se o nome do arquivo está correto e se ele existe na pasta.")
    sys.exit(1)

tqdm.write("[Pessoa 1] Imagem processada com sucesso.")


# ==========================================================
# ETAPA 2: Informação e Estrutura do Grafo
# ==========================================================
tqdm.write("\n[Transição] Extraindo dimensões da matriz...")

# A 'matriz_processada' é um array NumPy.
# A forma (shape) é (altura, largura, canais)
altura = matriz_processada.shape[0]
largura = matriz_processada.shape[1]
total_pixels = altura * largura

tqdm.write(f"Dimensões identificadas: Altura={altura}, Largura={largura}")
tqdm.write(f"Total de pixels (nós do grafo): {total_pixels}")

tqdm.write(f"\n[Pessoa 2] Criando o grafo de adjacência (8-vizinhos)...")
# Chama a sua função do arquivo 'grafo_basico.py'
lista_arestas = criar_grafo_adjacencia(altura, largura)
tqdm.write("[Pessoa 2] Grafo estrutural criado com sucesso.")


# ==========================================================
# ETAPA 3: Executar o cálculo de pesos
# ==========================================================

tqdm.write(f"\n[Pessoa 3] Calculando pesos (dist. Euclidiana) para {len(lista_arestas)} arestas...")
# Cria uma instância de barra de progresso para esta etapa
barra_pesos = tqdm(total=len(lista_arestas), desc="Calculando Pesos", unit=" arestas", leave=True, file=sys.stdout)

# Chama a função, criando a lista.
lista_arestas_com_pesos = calcular_pesos_arestas(matriz_processada, lista_arestas, barra_pesos)

# Fecha a barra de progresso
barra_pesos.close()

tqdm.write("[Pessoa 3] Cálculo de pesos concluído.")


# ==========================================================
# ETAPA 4: Executar o algoritmo de Kruskal
# ==========================================================

tqdm.write("\n[Pessoa 4] Executando o algoritmo de Kruskal (Árvore Geradora Mínima)...")
# Cria uma instância de barra de progresso para esta etapa
barra_kruskal = tqdm(total=len(lista_arestas_com_pesos), desc="Executando Kruskal", unit=" arestas", leave=True, file=sys.stdout)

# Chama a função, passando o objeto 'barra_kruskal' como argumento (aqui eh criado a lista  mst(peso, u, v)).
mst = kruskal_mst(lista_arestas_com_pesos, total_pixels, barra_kruskal)

# Fecha a barra de progresso
barra_kruskal.close()

tqdm.write(f"[Pessoa 4] Kruskal executado com sucesso.")

peso_total = sum([peso for peso, _, _ in mst])

# ==========================================================
tqdm.write("\n--- TESTE DE INTEGRAÇÃO (1 + 2 + 3 + 4) FINALIZADO COM SUCESSO ---")
# ==========================================================


# ==========================================================
# ETAPA 5: Resultados
# ==========================================================
tqdm.write("\n--- RESULTADOS DO TESTE ---")
tqdm.write(f"Total de Nós (Pixels): {total_pixels}")
tqdm.write(f"Total de Arestas (Conexões): {len(lista_arestas)}")

if total_pixels > 0:
    ratio = len(lista_arestas) / total_pixels
    tqdm.write(f"Taxa (Arestas / Nós): {ratio:.4f}")
    tqdm.write("(Para 8-vizinhos, este valor deve ser próximo de 4.0 para imagens grandes)")
    tqdm.write(f"Tamanho da MST: {len(mst)} arestas")
    tqdm.write(f"Peso total da MST: {peso_total:.6f}")