# Arquivo: visualizacao.py
# Autor: Pessoa 6
# Objetivo: Visualizar o resultado da segmentação, colorindo
#           cada segmento com sua cor L*a*b* média.

import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from scipy import ndimage as ndi
from tqdm import tqdm
import sys

def visualizar_segmentacao_lab(img_rgb_normalizada: np.ndarray, 
                               rotulos_map: np.ndarray,
                               salvar_arquivo: str = "resultado_segmentado_lab.png"):
    """
    Cria e exibe uma imagem segmentada.

    Calcula a cor L*a*b* média para cada segmento, pinta a imagem
    de saída com essas cores e, em seguida, converte o resultado
    final de volta para RGB para exibição.

    Args:
        img_rgb_normalizada: Matriz (H, W, 3) da Pessoa 1 (RGB, float [0, 1]).
        rotulos_map: Matriz (H, W) da Pessoa 5 (IDs de segmento, int).
        salvar_arquivo: Nome do arquivo para salvar a imagem final.
    """
    
    # Garantir que tqdm escreva corretamente no console
    tqdm_write = lambda s: tqdm.write(s, file=sys.stdout)
    
    tqdm_write("\n[Pessoa 6] Iniciando visualização...")
    
    # 1. Converter a imagem original de RGB para L*a*b*
    #    skimage lida diretamente com o formato float [0, 1]
    tqdm_write("[Pessoa 6] Convertendo imagem original para L*a*b*...")
    img_lab = color.rgb2lab(img_rgb_normalizada)

    # 2. Identificar todos os IDs de segmento únicos
    #    'rotulos_map' já deve conter IDs de 0 a N-1
    segmentos_unicos = np.unique(rotulos_map)
    num_segmentos = len(segmentos_unicos)
    tqdm_write(f"[Pessoa 6] Encontrados {num_segmentos} segmentos únicos.")

    # 3. Calcular a cor média (L, a, b) para cada segmento
    #    Usar `scipy.ndimage.mean` é MUITO mais rápido do que um loop
    #    Ele calcula a média de 'img_lab[canal]' para cada 'label' em 'rotulos_map'
    tqdm_write("[Pessoa 6] Calculando cores L*a*b* médias para cada segmento...")
    avg_l = ndi.mean(img_lab[:, :, 0], labels=rotulos_map, index=segmentos_unicos)
    avg_a = ndi.mean(img_lab[:, :, 1], labels=rotulos_map, index=segmentos_unicos)
    avg_b = ndi.mean(img_lab[:, :, 2], labels=rotulos_map, index=segmentos_unicos)

    # Cria uma "tabela de consulta" de cores médias
    # (num_segmentos, 3)
    cores_medias_lab = np.stack((avg_l, avg_a, avg_b), axis=1)

    # 4. Criar a nova imagem (de saída) em L*a*b*
    tqdm_write("[Pessoa 6] Pintando a imagem de saída com as cores médias...")
    img_segmentada_lab = np.zeros_like(img_lab)
    
    # Itera sobre os segmentos e "pinta" a imagem de saída
    # Este loop é rápido, pois itera 'num_segmentos', não 'num_pixels'
    for i, segmento_id in enumerate(tqdm(segmentos_unicos, 
                                        desc="Pintando Segmentos", 
                                        file=sys.stdout, 
                                        ncols=80)):
        
        # Pega a cor média para este segmento
        cor_media = cores_medias_lab[i]
        
        # Pinta todos os pixels pertencentes a este segmento
        img_segmentada_lab[rotulos_map == segmento_id] = cor_media

    # 5. Converter a imagem segmentada de L*a*b* de volta para RGB
    tqdm_write("[Pessoa 6] Convertendo imagem final de L*a*b* para RGB...")
    img_segmentada_rgb = color.lab2rgb(img_segmentada_lab)
    
    # Garante que os valores estejam no intervalo [0, 1]
    # (Conversões de gamut podem gerar valores ligeiramente fora)
    img_segmentada_rgb = np.clip(img_segmentada_rgb, 0, 1)

    # 6. Usar Matplotlib para exibir lado a lado
    tqdm_write("[Pessoa 6] Exibindo resultado...")
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb_normalizada)
    plt.title("Imagem Original (Pessoa 1)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_segmentada_rgb)
    plt.title(f"Segmentação L*a*b* Média (Pessoa 6)\n{num_segmentos} regiões")
    plt.axis('off')
    
    plt.tight_layout()
    
    # 7. Salvar a imagem final
    if salvar_arquivo:
        try:
            # Salva a imagem (matplotlib lida bem com float [0, 1])
            plt.savefig(salvar_arquivo, dpi=300)
            tqdm_write(f"[Pessoa 6] Imagem segmentada salva em '{salvar_arquivo}'")
        except Exception as e:
            tqdm_write(f"AVISO: Não foi possível salvar a imagem. Erro: {e}")

    plt.show()

    return img_segmentada_rgb

# --- Bloco de Teste ---
if __name__ == "__main__":
    
    tqdm.write("Iniciando teste local de visualizacao.py (Pessoa 6)...")

    # 1. Simular a entrada da Pessoa 1 (img_rgb_normalizada)
    #    Uma imagem 200x200 com 4 quadrantes de cores
    altura, largura = 200, 200
    img_original_simulada = np.zeros((altura, largura, 3), dtype=float)
    
    # Canto superior esquerdo: Vermelho
    img_original_simulada[0:100, 0:100] = [1.0, 0.1, 0.1]
    # Canto superior direito: Verde
    img_original_simulada[0:100, 100:200] = [0.1, 0.8, 0.1]
    # Canto inferior esquerdo: Azul
    img_original_simulada[100:200, 0:100] = [0.1, 0.1, 0.9]
    # Canto inferior direito: Amarelo
    img_original_simulada[100:200, 100:200] = [0.9, 0.9, 0.2]
    
    # Adiciona um pouco de ruído para simular uma imagem real
    ruido = np.random.rand(altura, largura, 3) * 0.05
    img_original_simulada = np.clip(img_original_simulada + ruido, 0, 1)

    # 2. Simular a entrada da Pessoa 5 (rotulos_map)
    #    Vamos supor que a segmentação foi perfeita
    rotulos_simulados = np.zeros((altura, largura), dtype=int)
    rotulos_simulados[0:100, 0:100] = 0  # Região Vermelha
    rotulos_simulados[0:100, 100:200] = 1  # Região Verde
    rotulos_simulados[100:200, 0:100] = 2  # Região Azul
    rotulos_simulados[100:200, 100:200] = 3  # Região Amarela

    # 3. Executar a função da Pessoa 6
    visualizar_segmentacao_lab(img_original_simulada, rotulos_simulados)

    tqdm.write("\nTeste local de visualizacao.py concluído.")
    tqdm.write("Verifique a janela do Matplotlib.")
    tqdm.write("A imagem da direita deve ter 4 blocos de cores sólidas,")
    tqdm.write("representando a cor média L*a*b* de cada quadrante.")