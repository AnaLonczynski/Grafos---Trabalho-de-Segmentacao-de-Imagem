
# ------------------------------------------------------------------------------
#|Calcula a cor L*a*b* média para cada segmento, pinta a imagem                 |
#|de saída com essas cores e, em seguida, converte o resultad                   |
#|final de volta para RGB para exibição.                                        |
#|   Args:                                                                      |
#|        img_rgb_normalizada: Matriz (H, W, 3) da Pessoa 1 (RGB, float [0, 1]).|
#|        rotulos_map: Matriz (H, W) da Pessoa 5 (IDs de segmento, int).        |
#|        salvar_arquivo: Nome do arquivo para salvar a imagem final.           |
# ------------------------------------------------------------------------------
 
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
    
    tqdm_write("\n Iniciando visualização...")
    
    img_lab = color.rgb2lab(img_rgb_normalizada)

    #     Identificar todos os IDs de segmento únicos
    #    'rotulos_map' já deve conter IDs de 0 a N-1
    segmentos_unicos = np.unique(rotulos_map)
    num_segmentos = len(segmentos_unicos)
    tqdm_write(f" Encontrados {num_segmentos} segmentos únicos.")

    #    Usar `scipy.ndimage.mean` é MUITO mais rápido do que um loop
    #    Ele calcula a média de 'img_lab[canal]' para cada 'label' em 'rotulos_map'
    tqdm_write("[ Calculando cores médias para cada segmento...")
    avg_l = ndi.mean(img_lab[:, :, 0], labels=rotulos_map, index=segmentos_unicos)
    avg_a = ndi.mean(img_lab[:, :, 1], labels=rotulos_map, index=segmentos_unicos)
    avg_b = ndi.mean(img_lab[:, :, 2], labels=rotulos_map, index=segmentos_unicos)

    cores_medias_lab = np.stack((avg_l, avg_a, avg_b), axis=1)

    tqdm_write(" Pintando a imagem de saída com as cores médias...")
    img_segmentada_lab = np.zeros_like(img_lab)
    
    for i, segmento_id in enumerate(tqdm(segmentos_unicos, 
                                        desc="Pintando Segmentos", 
                                        file=sys.stdout, 
                                        ncols=80)):
        
        cor_media = cores_medias_lab[i]
        
        img_segmentada_lab[rotulos_map == segmento_id] = cor_media

    img_segmentada_rgb = color.lab2rgb(img_segmentada_lab)
    
    # Garante que os valores estejam no intervalo [0, 1]
    # (Conversões de gamut podem gerar valores ligeiramente fora)
    img_segmentada_rgb = np.clip(img_segmentada_rgb, 0, 1)

    # 6. Usar Matplotlib para exibir lado a lado
    tqdm_write(" Exibindo resultado...")
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb_normalizada)
    plt.title("Imagem Original")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_segmentada_rgb)
    plt.title(f"Segmentação RGB Média\n{num_segmentos} regiões")
    plt.axis('off')
    
    plt.tight_layout()
    
    # 7. Salvar a imagem final
    if salvar_arquivo:
        try:
            # Salva a imagem (matplotlib lida bem com float [0, 1])
            plt.savefig(salvar_arquivo, dpi=300)
            tqdm_write(f" Imagem segmentada salva em '{salvar_arquivo}'")
        except Exception as e:
            tqdm_write(f"AVISO: Não foi possível salvar a imagem. Erro: {e}")

    plt.show()

    return img_segmentada_rgb