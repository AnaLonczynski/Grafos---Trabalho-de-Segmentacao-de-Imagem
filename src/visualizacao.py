import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from scipy import ndimage as ndi
from tqdm import tqdm
import sys

def visualizar_segmentacao_lab(img_rgb_normalizada: np.ndarray, 
                               rotulos_map: np.ndarray,
                               salvar_arquivo: str = "resultado_segmentado_lab.png"):
    
    tqdm_write = lambda s: tqdm.write(s, file=sys.stdout)
    
    tqdm_write("\nIniciando visualização e colorização...")
    
    # Converter para LAB 
    img_lab = color.rgb2lab(img_rgb_normalizada)

    # Identificar segmentos
    segmentos_unicos = np.unique(rotulos_map)
    num_segmentos = len(segmentos_unicos)
    tqdm_write(f"Encontrados {num_segmentos} segmentos únicos.")

    # Calcular a cor média 
    tqdm_write("Calculando cores médias (L*a*b*)...")
    
    avg_l = ndi.mean(img_lab[:, :, 0], labels=rotulos_map, index=segmentos_unicos)
    avg_a = ndi.mean(img_lab[:, :, 1], labels=rotulos_map, index=segmentos_unicos)
    avg_b = ndi.mean(img_lab[:, :, 2], labels=rotulos_map, index=segmentos_unicos)

    # Pintar a imagem 
    tqdm_write(" Aplicando cores aos segmentos (Vetorizado)...")
    
    # maior ID para criar tabelas de busca
    max_id = np.max(rotulos_map)
    
    # ID do Segmento -> Valor de Cor
    lookup_l = np.zeros(max_id + 1)
    lookup_a = np.zeros(max_id + 1)
    lookup_b = np.zeros(max_id + 1)
    
    lookup_l[segmentos_unicos] = avg_l
    lookup_a[segmentos_unicos] = avg_a
    lookup_b[segmentos_unicos] = avg_b
    
    final_l = lookup_l[rotulos_map]
    final_a = lookup_a[rotulos_map]
    final_b = lookup_b[rotulos_map]
    
    img_segmentada_lab = np.stack((final_l, final_a, final_b), axis=-1)

    # Converter de volta para RGB
    img_segmentada_rgb = color.lab2rgb(img_segmentada_lab)

    img_segmentada_rgb = np.clip(img_segmentada_rgb, 0, 1)

    # 6. Exibir e Salvar
    tqdm_write(" Gerando imagem final...")
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb_normalizada)
    plt.title("Imagem Original")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_segmentada_rgb)
    plt.title(f"Resultado Segmentado\n({num_segmentos} regiões)")
    plt.axis('off')
    
    plt.tight_layout()
    
    if salvar_arquivo:
        try:
            plt.savefig(salvar_arquivo, dpi=300)
            tqdm_write(f" Sucesso! Imagem salva em '{salvar_arquivo}'")
        except Exception as e:
            tqdm_write(f"AVISO: Erro ao salvar imagem: {e}")

    return img_segmentada_rgb