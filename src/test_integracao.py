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
import os  # <-- ADICIONE ESTA LINHA

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# 1. Importar as funções dos outros arquivos
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

# --- INÍCIO DO TESTE ---

# Defina o nome da imagem a ser testada
NOME_ARQUIVO_TESTE = "totoro.jpg"

print("--- INICIANDO TESTE DE INTEGRAÇÃO (PESSOA 1 + PESSOA 2) ---")

# ==========================================================
# ETAPA 1: Executar o código da Pessoa 1
# ==========================================================
print(f"\n[Pessoa 1] Processando a imagem: '{NOME_ARQUIVO_TESTE}'...")

# Chama a função do arquivo 'preprocs.py'
matriz_processada = preprocessar_imagem(NOME_ARQUIVO_TESTE)

# Checagem de erro
if matriz_processada is None:
    print(f"ERRO FATAL: Falha ao carregar '{NOME_ARQUIVO_TESTE}'.")
    print("Verifique se o nome do arquivo está correto e se ele existe na pasta.")
    sys.exit(1)

print("[Pessoa 1] Imagem processada com sucesso.")


# ==========================================================
# ETAPA 2: Passar a informação para a Pessoa 2
# ==========================================================
print("\n[Transição] Extraindo dimensões da matriz...")

# A 'matriz_processada' é um array NumPy.
# A forma (shape) é (altura, largura, canais)
altura = matriz_processada.shape[0]
largura = matriz_processada.shape[1]
total_pixels = altura * largura

print(f"Dimensões identificadas: Altura={altura}, Largura={largura}")
print(f"Total de pixels (nós do grafo): {total_pixels}")


# ==========================================================
# ETAPA 3: Executar o seu código (Pessoa 2)
# ==========================================================
print(f"\n[Pessoa 2] Criando o grafo de adjacência (8-vizinhos)...")

# Chama a sua função do arquivo 'grafo_basico.py'
# Esta é a sua "Saída"
lista_arestas = criar_grafo_adjacencia(altura, largura)

print("[Pessoa 2] Grafo estrutural criado com sucesso.")

# ==========================================================
# ETAPA 4: Verificar os resultados
# ==========================================================
print("\n--- RESULTADOS DO TESTE ---")
print(f"Total de Nós (Pixels): {total_pixels}")
print(f"Total de Arestas (Conexões): {len(lista_arestas)}")

if total_pixels > 0:
    # Para um grafo 8-vizinhanças, o número de arestas se aproxima de 4 * N
    # (onde N é o total de pixels), pois cada pixel se conecta a 4 vizinhos
    # (os outros 4 são cobertos por conexões "de volta")
    ratio = len(lista_arestas) / total_pixels
    print(f"Taxa (Arestas / Nós): {ratio:.4f}")
    print("(Para 8-vizinhos, este valor deve ser próximo de 4.0 para imagens grandes)")

# Imprime uma amostra para ver se parece correto
print("\nAmostra das primeiras 10 arestas geradas:")
print(lista_arestas[:10])

print("\nAmostra das últimas 10 arestas geradas:")
print(lista_arestas[-10:])

print("\n--- TESTE DE INTEGRAÇÃO CONCLUÍDO ---")