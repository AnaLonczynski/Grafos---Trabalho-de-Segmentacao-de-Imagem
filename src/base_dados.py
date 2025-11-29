#nome: matheusEduardoCamposSoares - 837435

"""
base_dados.py
Unifica:
 - leitura e normalização de imagem
 - construção de grafo DIRECIONADO (4 ou 8 vizinhos)
 - cálculo de pesos entre pixels (distância de cor)
 - salvamento em .npz e .csv
 - funções simples de inspeção/visualização

Uso:
  python base_dados.py --img imagens/exemplo.jpg --out dados/edges_saida --maxsize 300 --neigh 4
"""

from typing import Tuple, List, Dict
import numpy as np
import cv2
import os
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------
# Utilitários de ID <-> coordenada
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
# Construção do grafo DIRECIONADO
# -----------------------
def gerar_arestas_direcionadas(altura: int, largura: int, vizinhanca: str = "4") -> List[Tuple[int,int]]:
    """
    Gera lista de arestas direcionadas (u,v) sem pesos.
    vizinhanca: "4" ou "8".
    Nota: cada par de vizinhos será representado em ambos os sentidos porque a função varre todos os pixels.
    """
    arestas: List[Tuple[int,int]] = []
    # offsets para vizinhança 4
    offsets_4 = [(0,1),(1,0),(0,-1),(-1,0)]
    # offsets adicionais para vizinhança 8 (diagonais)
    offsets_8 = offsets_4 + [(-1,-1),(-1,1),(1,-1),(1,1)]
    offsets = offsets_4 if vizinhanca == "4" else offsets_8  # mudar de "4" para "8" para aumentar quantidade de ligações

    for linha in range(altura):
        for coluna in range(largura):
            u = coord_para_id(linha, coluna, largura)
            for dr, dc in offsets:
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
    # usar tqdm para ver progresso em imagens maiores
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
# Salvamento / Leitura
# -----------------------
def salvar_arestas_npz(caminho_saida: str, altura: int, largura: int, pesos_arestas: List[Tuple[int,int,float]], metadados: Dict = None):
    """
    Salva arrays u, v, w e metadados em arquivo .npz comprimido.
    """
    u_arr = np.array([t[0] for t in pesos_arestas], dtype=np.int32)
    v_arr = np.array([t[1] for t in pesos_arestas], dtype=np.int32)
    w_arr = np.array([t[2] for t in pesos_arestas], dtype=np.float32)
    meta = metadados.copy() if metadados else {}
    meta.update({"altura": altura, "largura": largura})
    # meta salvo como objeto para manter dicionário
    np.savez_compressed(caminho_saida + ".npz", u=u_arr, v=v_arr, w=w_arr, meta=np.array([meta], dtype=object))
    print(f"Salvo {len(u_arr)} arestas em {caminho_saida}.npz")

def carregar_arestas_npz(caminho_npz: str) -> Tuple[int,int,List[Tuple[int,int,float]],Dict]:
    """
    Carrega arquivo .npz gerado por salvar_arestas_npz e retorna (altura, largura, lista_de_(u,v,w), meta)
    """
    d = np.load(caminho_npz, allow_pickle=True)
    u_arr = d["u"].astype(np.int32)
    v_arr = d["v"].astype(np.int32)
    w_arr = d["w"].astype(np.float32)
    meta = d["meta"][0].item() if "meta" in d else {}
    altura = meta.get("altura")
    largura = meta.get("largura")
    lista_pesos = list(zip(u_arr.tolist(), v_arr.tolist(), w_arr.tolist()))
    return altura, largura, lista_pesos, meta

def salvar_arestas_csv(caminho_csv: str, pesos_arestas: List[Tuple[int,int,float]]):
    """
    Salva arestas em CSV com cabeçalho u,v,w — útil para inspeção humana (Excel/Sheets).
    """
    with open(caminho_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["u","v","w"])
        for u, v, w in pesos_arestas:
            writer.writerow([u, v, f"{w:.6f}"])
    print(f"CSV salvo em {caminho_csv}")

# -----------------------
# Inspeção rápida / Visualizações
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

def plot_histograma_pesos(pesos_arestas: List[Tuple[int,int,float]], numero_bins: int = 50, salvar_caminho: str = None, exibir: bool = True):
    pesos = np.array([t[2] for t in pesos_arestas], dtype=np.float32)
    plt.figure(figsize=(6,4))
    plt.hist(pesos, bins=numero_bins)
    plt.title("Histograma de pesos (distância de cor)")
    plt.xlabel("Peso")
    plt.ylabel("Frequência")
    if salvar_caminho:
        plt.savefig(salvar_caminho, bbox_inches="tight", dpi=150)
        print(f"Histograma salvo em {salvar_caminho}")
    if exibir:
        plt.show()
    else:
        plt.close()

def desenhar_overlay_grafo(img_rgb_normalizada: np.ndarray, lista_arestas: List[Tuple[int,int,float]], max_arestas: int = 1000):
    """
    Desenha uma amostra das arestas sobre a imagem. Só usar com imagens pequenas/reduzidas.
    """
    try:
        import networkx as nx
    except Exception:
        print("networkx não instalado — pip install networkx para overlay do grafo.")
        return
    altura, largura = img_rgb_normalizada.shape[:2]
    G = nx.DiGraph()
    for linha in range(altura):
        for coluna in range(largura):
            idx = coord_para_id(linha, coluna, largura)
            G.add_node(idx, pos=(coluna, altura - 1 - linha))
    amostra_arestas = [(u, v) for (u, v, _) in lista_arestas[:max_arestas]]
    for u, v in amostra_arestas:
        G.add_edge(u, v)
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(6,6))
    plt.imshow(img_rgb_normalizada)
    nx.draw_networkx_nodes(G, pos, node_size=5)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=6, width=0.5)
    plt.title("Overlay do grafo (amostra)")
    plt.axis('off')
    plt.show()

# -----------------------
# Pipeline principal (tudo integrado)
# -----------------------
def pipeline_unificado(caminho_imagem: str,
                       caminho_saida_base: str,
                       max_lado: int = 200,
                       vizinhanca: str = "4",
                       gerar_plots: bool = True) -> Tuple[np.ndarray, List[Tuple[int,int,float]]]:
    """
    Executa pipeline completo: leitura -> gerar arestas direcionadas -> calcular pesos -> salvar .npz e .csv -> inspeção.
    Retorna (imagem_normalizada, lista_de_(u,v,w)).
    """
    img = carregar_imagem_rgb_normalizada(caminho_imagem, max_lado)
    altura, largura = img.shape[:2]
    print(f"Imagem carregada {os.path.basename(caminho_imagem)} — {largura}x{altura}")
    arestas = gerar_arestas_direcionadas(altura, largura, vizinhanca)
    print(f"Arestas direcionadas geradas: {len(arestas)}")
    pesos = calcular_pesos_por_cor(img, arestas)

    # salvar .npz e .csv
    metadados = {"origem": os.path.basename(caminho_imagem), "vizinhanca": vizinhanca}
    salvar_arestas_npz(caminho_saida_base, altura, largura, pesos, metadados)
    salvar_arestas_csv(caminho_saida_base + ".csv", pesos)

    # inspeção
    estatisticas_rapidas(altura, largura, pesos)
    if gerar_plots:
        plot_histograma_pesos(pesos, numero_bins=50, salvar_caminho=caminho_saida_base + "_hist.png", exibir=True)
        if max(altura, largura) <= 150:
            desenhar_overlay_grafo(img, pesos, max_arestas=500)
    return img, pesos

# -----------------------
# Execução via CLI
# -----------------------
if __name__ == "__main__":
    # Caminho da imagem que será processada
    caminho_imagem = "../jiji.jpg"  
    # Mudar para o caminho da sua imagem, por exemplo:
    # "minhas_fotos/foto1.jpg"

    # Prefixo dos arquivos de saída (vai gerar .npz e .csv)
    caminho_saida = "dados/resultado_teste"
    # Mudar o nome se quiser separar os resultados

    # Tamanho máximo permitido para o lado maior da imagem
    max_lado_imagem = None  
    # Aumentar para 300–400 deixa mais detalhes, mas pesa mais na memória

    # Tipo de vizinhança: "4" ou "8"
    vizinhanca_pixels = "4"
    # Trocar para "8" aumenta a quantidade de ligações no grafo

    # Gerar (True) ou não gerar (False) as imagens de visualização
    gerar_plots = True  

    # Chamada direta do pipeline
    pipeline_unificado(
        caminho_imagem,
        caminho_saida,
        max_lado=max_lado_imagem,
        vizinhanca=vizinhanca_pixels,
        gerar_plots=gerar_plots
    )
