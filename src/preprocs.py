# PASSO 1: Importar as bibliotecas necessárias
import cv2  # Esta é a biblioteca OpenCV
import numpy as np # Esta é a biblioteca NumPy

def preprocessar_imagem(caminho_imagem, aplicar_blur=True, kernel_blur=(5, 5)):
    """
    Lê uma imagem, converte para o espaço de cor L*a*b*, aplica um blur
    opcional e normaliza os canais para o intervalo [0, 1].

    Parâmetros:
    - caminho_imagem (str): O caminho para o arquivo da imagem.
    - aplicar_blur (bool): Se True, aplica um Gaussian Blur.
    - kernel_blur (tuple): O tamanho do kernel para o blur (ex: (5, 5)).

    Retorna:
    - np.ndarray: A matriz 3D (imagem) processada e pronta para o grafo.
                   Retorna None se a imagem não puder ser lida.
    """

    # ----------------------------------------------------
    # Tarefa 1: Ler a imagem
    # ----------------------------------------------------
    # cv2.imread lê a imagem. Por padrão, ele a lê no formato BGR (Blue, Green, Red).
    imagem_bgr = cv2.imread(caminho_imagem)

    # Checagem de segurança: se a imagem não foi encontrada, imagem_bgr será 'None'
    if imagem_bgr is None:
        print(f"Erro: Não foi possível ler a imagem em '{caminho_imagem}'")
        return None

    # ----------------------------------------------------
    # Tarefa "Extra" (Decisão do Grupo): Converter para L*a*b*
    # ----------------------------------------------------
    # Esta é a nossa estratégia para trabalhar com cores de forma eficaz.
    # Convertemos do espaço BGR (padrão do OpenCV) para o L*a*b*.
    imagem_lab = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2Lab)

    # ----------------------------------------------------
    # Tarefa 3: Normalizar os valores (0-1)
    # ----------------------------------------------------
    # Os canais L*a*b* têm intervalos diferentes:
    # L: 0 a 100
    # a: -128 a 127
    # b: -128 a 127
    # Vamos normalizar todos para [0, 1] para que tenham pesos iguais.

    imagem_lab = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L_true = imagem_lab[:,:,0] * (100.0 / 255.0)
    L_norm = L_true / 100.0
    a_norm = imagem_lab[:,:,1] / 255.0
    b_norm = imagem_lab[:,:,2] / 255.0
    imagem_normalizada = np.stack([L_norm, a_norm, b_norm], axis=2).astype(np.float32)


    # ----------------------------------------------------
    # Tarefa 4: Aplicar blur opcional
    # ----------------------------------------------------
    imagem_final = imagem_normalizada
    if aplicar_blur:
        # O GaussianBlur suaviza a imagem, removendo ruídos.
        # Isso ajuda o algoritmo de segmentação a não criar
        # pequenos segmentos inúteis por causa de um único pixel diferente.
        imagem_final = cv2.GaussianBlur(imagem_normalizada, kernel_blur, 0)

    # ----------------------------------------------------
    # Tarefa 5: Retornar a matriz pronta
    # ----------------------------------------------------
    # O resultado é uma matriz 3D (Altura x Largura x 3),
    # onde os 3 canais são L*, a*, b* normalizados.
    return imagem_final

# FFFFFF

imagem = preprocessar_imagem("totoro_rebaixado.jpg")

if imagem is not None:
    print(imagem.shape)  # Exibe o formato (altura, largura, canais)
    print(imagem.dtype)  # Tipo dos dados (provavelmente float32)
    print(imagem.min(), imagem.max())  # Intervalo de valores (esperado: 0.0 a 1.0)

    # Mostra um pequeno pedaço da matriz (para não imprimir tudo)
    print(imagem[0:10, 0:10, :])  # Mostra os 3 primeiros pixels (3x3)