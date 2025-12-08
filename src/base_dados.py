from typing import Tuple, List, Dict
import numpy as np
import cv2
import os
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------
# ID <-> coordenada
# -----------------------
def coord_para_id(linha: int, coluna: int, largura: int) -> int:
    return linha * largura + coluna

def id_para_coord(indice: int, largura: int) -> Tuple[int, int]:
    return divmod(indice, largura)  # devolve (linha, coluna)

# -----------------------
# Leitura e normalização de imagem
# -----------------------
def carregar_imagem_rgb_normalizada(caminho_imagem: str, max_lado: int = None) -> np.ndarray:
    """
    Carrega imagem e retorna array RGB float32 em [0,1].
    max_lado: se definido, redimensiona mantendo proporção para max(width,height) <= max_lado.
    """
    if not os.path.exists(caminho_imagem):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_imagem}")
    img_bgr = cv2.imread(caminho_imagem, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Erro ao carregar imagem com cv2.imread")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if max_lado is not None:
        altura, largura = img_rgb.shape[:2]
        escala = min(1.0, max_lado / max(altura, largura))
        if escala < 1.0:
            nova_largura = int(largura * escala)
            nova_altura = int(altura * escala)
            img_rgb = cv2.resize(img_rgb, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
    img_float = img_rgb.astype(np.float32) / 255.0
    return img_float

# -----------------------
# Construção do grafo DIRECIONADO (com a restrição de arestas)
# -----------------------
def gerar_arestas_direcionadas(altura: int, largura: int, vizinhanca: str = "4") -> List[Tuple[int,int]]:
    """
    Gera lista de arestas direcionadas (u,v) **acíclicas**.
    As direções permitidas são: Baixo, Direita e Diagonal Direita Baixo.
    
    vizinhanca: "4" (Baixo e Direita) ou "8" (Baixo, Direita, Diagonais).
    """
    arestas: List[Tuple[int,int]] = []
    
    if vizinhanca == "4":
        # Baixo (1, 0) e Direita (0, 1)
        offsets_permitidos = [(1, 0), (0, 1)]
    elif vizinhanca == "8":
        # Baixo (1, 0), Direita (0, 1) e Diagonal Direita Baixo (1, 1)
        offsets_permitidos = [(1, 0), (0, 1), (1, 1)] 
    else:
         raise ValueError("Vizinhanca deve ser '4' (Baixo/Direita) ou '8' (Baixo/Direita/Diag. Baixo Direita)")

    for linha in range(altura):
        for coluna in range(largura):
            u = coord_para_id(linha, coluna, largura)
            
            # Conecta de u para os vizinhos permitidos
            for dr, dc in offsets_permitidos:
                nl, nc = linha + dr, coluna + dc
                
                if 0 <= nl < altura and 0 <= nc < largura:
                    v = coord_para_id(nl, nc, largura)
                    arestas.append((u, v))
                    
    return arestas

# -----------------------
# Cálculo de pesos
# -----------------------
def calcular_pesos_por_cor(img_rgb_normalizada: np.ndarray, lista_arestas: List[Tuple[int,int]], metrica: str = "euclidiana") -> List[Tuple[int,int,float]]:
    """
    Para cada aresta (u,v), calcula peso w = distância entre cor de u e v.
    Retorna lista de (u, v, w).
    """

    altura, largura = img_rgb_normalizada.shape[:2]

    def cor_por_id(idx: int) -> np.ndarray:
        l, c = id_para_coord(idx, largura)
        return img_rgb_normalizada[l, c]  # vetor [R,G,B]

    pesos: List[Tuple[int,int,float]] = []

    for (u, v) in tqdm(lista_arestas, desc="Calculando pesos"):
        cor_u = cor_por_id(u)
        cor_v = cor_por_id(v)
        if metrica == "euclidiana" or metrica == "euclidiana_rgb":
            w = float(np.linalg.norm(cor_u - cor_v))
        else:
            raise NotImplementedError("Apenas 'euclidiana' implementado")
        pesos.append((u, v, w))
    return pesos

# -----------------------
# Inspeção rápida
# -----------------------
def estatisticas_rapidas(altura: int, largura: int, pesos_arestas: List[Tuple[int,int,float]]):
    """
    Imprime informações básicas para verificação.
    """
    n_nos = altura * largura
    n_arestas = len(pesos_arestas)
    print("=== ESTATÍSTICAS RÁPIDAS ===")
    print(f"Pixels (nós): {n_nos}")
    print(f"Arestas direcionadas: {n_arestas}")
    print(f"Arestas por nó (média): {n_arestas / n_nos:.2f}")
    print("Amostra de até 10 arestas (u, v, w):")
    for t in pesos_arestas[:10]:
        print(t)
    # checagem de contagem com fórmula para 4/8
    n_undirected_4 = altura * (largura - 1) + (altura - 1) * largura
    n_undirected_8 = n_undirected_4 + 2 * (altura - 1) * (largura - 1)
    print(f"Estimativa (não-direcionado) 4-neigh: {n_undirected_4}, 8-neigh: {n_undirected_8}")
    print(f"Estimativa (direcionado) 4-neigh: {2 * n_undirected_4}, 8-neigh: {2 * n_undirected_8}")
    print("============================")

# -----------------------
# Integração
# -----------------------
def pipeline_unificado(caminho_imagem: str,
                       caminho_saida_base: str,
                       max_lado: int = 200,
                       vizinhanca: str = "4",
                       gerar_plots: bool = True) -> Tuple[np.ndarray, List[Tuple[int,int,float]]]:
    """
    leitura -> gerar arestas direcionadas -> calcular pesos
    Retorna (imagem_normalizada, lista_de_(u,v,w)).
    """
    img = carregar_imagem_rgb_normalizada(caminho_imagem, max_lado)
    altura, largura = img.shape[:2]
    print(f"Imagem carregada {os.path.basename(caminho_imagem)} — {largura}x{altura}")
    arestas = gerar_arestas_direcionadas(altura, largura, vizinhanca)
    print(f"Arestas direcionadas geradas: {len(arestas)}")
    pesos = calcular_pesos_por_cor(img, arestas)

    # inspeção
    estatisticas_rapidas(altura, largura, pesos)
    if gerar_plots:
        if max(altura, largura) <= 150:
            desenhar_overlay_grafo(img, pesos, max_arestas=500)
    return img, pesos
