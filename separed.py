import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_question_blocks(path, 
                          h_start_frac=0.2, h_end_frac=0.95,
                          min_block_height=100,
                          col_thresh_frac=0.03,
                          row_thresh_frac=0.05):
    """
    Detecta blocos de questões em uma prova, ignorando cabeçalho e rodapé.
    
    Parâmetros:
    - h_start_frac: fração da altura onde começar a procurar (evita cabeçalho)
    - h_end_frac: fração da altura onde parar de procurar (evita rodapé)
    - min_block_height: altura mínima para considerar um bloco válido
    - col_thresh_frac: limiar para detectar colunas com conteúdo
    - row_thresh_frac: limiar para detectar linhas com conteúdo
    """
    
    # 1) Carrega e converte
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    
    # 2) Define região de interesse (ignora cabeçalho e rodapé)
    y_start = int(H * h_start_frac)
    y_end = int(H * h_end_frac)
    
    # 3) Aplica threshold adaptativo na região de interesse
    roi_gray = gray[y_start:y_end, :]
    th = cv2.adaptiveThreshold(roi_gray, 255,
                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 15, 10)
    
    # 4) Projeção horizontal para encontrar áreas com texto/questões
    row_sum = np.sum(th > 0, axis=1)
    row_thresh = W * row_thresh_frac
    
    # Encontra linhas com conteúdo significativo
    content_rows = np.where(row_sum > row_thresh)[0]
    
    if len(content_rows) == 0:
        print("Nenhuma linha com conteúdo encontrada!")
        return None, None
    
    # 5) Encontra o bloco principal de conteúdo
    # Agrupa linhas consecutivas para encontrar blocos contínuos
    diff = np.diff(content_rows)
    gaps = np.where(diff > 10)[0]  # gaps maiores que 10 linhas
    
    if len(gaps) > 0:
        # Pega o maior bloco contínuo
        blocks = []
        start_idx = 0
        for gap_idx in gaps:
            end_idx = gap_idx
            block_start = content_rows[start_idx]
            block_end = content_rows[end_idx]
            blocks.append((block_start, block_end, block_end - block_start))
            start_idx = gap_idx + 1
        
        # Adiciona último bloco
        block_start = content_rows[start_idx]
        block_end = content_rows[-1]
        blocks.append((block_start, block_end, block_end - block_start))
        
        # Pega o bloco com maior altura
        blocks.sort(key=lambda x: x[2], reverse=True)
        main_block_start, main_block_end, _ = blocks[0]
    else:
        main_block_start = content_rows[0]
        main_block_end = content_rows[-1]
    
    # Verifica se o bloco tem altura mínima
    if (main_block_end - main_block_start) < min_block_height:
        print(f"Bloco muito pequeno: {main_block_end - main_block_start} pixels")
        return None, None
    
    # 6) Converte coordenadas de volta para a imagem original
    y0 = y_start + main_block_start
    y1 = y_start + main_block_end
    
    # 7) Projeção vertical para encontrar limites horizontais
    block_th = th[main_block_start:main_block_end+1, :]
    col_sum = np.sum(block_th > 0, axis=0)
    col_thresh = (main_block_end - main_block_start) * col_thresh_frac
    
    valid_cols = np.where(col_sum > col_thresh)[0]
    
    if len(valid_cols) == 0:
        print("Nenhuma coluna com conteúdo encontrada!")
        return None, None
    
    x0, x1 = valid_cols[0], valid_cols[-1]
    
    # 8) Faz o crop do bloco de questões
    crop = img[y0:y1+1, x0:x1+1]
    h_crop, w_crop = crop.shape[:2]
    
    # 9) Detecta se há duas colunas
    # Analisa a distribuição vertical de conteúdo
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_th = cv2.adaptiveThreshold(crop_gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    
    col_projection = np.sum(crop_th > 0, axis=0)
    
    # Procura por vale no meio (separação entre colunas)
    mid_region = w_crop // 3
    mid_start = w_crop // 2 - mid_region // 2
    mid_end = w_crop // 2 + mid_region // 2
    
    mid_values = col_projection[mid_start:mid_end]
    min_mid = np.min(mid_values)
    max_sides = max(np.max(col_projection[:mid_start]), 
                   np.max(col_projection[mid_end:]))
    
    # Se há um vale significativo no meio, provavelmente são duas colunas
    if max_sides > 0 and min_mid < max_sides * 0.3:
        # Encontra o ponto de divisão mais preciso
        min_idx = np.argmin(mid_values) + mid_start
        left = crop[:, :min_idx]
        right = crop[:, min_idx:]
        two_columns = True
    else:
        # Divide simples no meio
        mid = w_crop // 2
        left = crop[:, :mid]
        right = crop[:, mid:]
        two_columns = False
    
    # 10) Visualização
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Primeira linha: imagem original com ROI marcada
    axes[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0,0].axhline(y=y_start, color='red', linestyle='--', alpha=0.7)
    axes[0,0].axhline(y=y_end, color='red', linestyle='--', alpha=0.7)
    axes[0,0].axhline(y=y0, color='green', linestyle='-', linewidth=2)
    axes[0,0].axhline(y=y1, color='green', linestyle='-', linewidth=2)
    axes[0,0].axvline(x=x0, color='green', linestyle='-', linewidth=2)
    axes[0,0].axvline(x=x1, color='green', linestyle='-', linewidth=2)
    axes[0,0].set_title("Imagem original + detecção")
    axes[0,0].axis("off")
    
    # Projeções
    axes[0,1].plot(row_sum)
    axes[0,1].axhline(y=row_thresh, color='red', linestyle='--')
    axes[0,1].set_title("Projeção horizontal")
    axes[0,1].set_xlabel("Linha")
    axes[0,1].set_ylabel("Pixels preenchidos")
    
    axes[0,2].plot(col_projection)
    axes[0,2].set_title("Projeção vertical do bloco")
    axes[0,2].set_xlabel("Coluna")
    axes[0,2].set_ylabel("Pixels preenchidos")
    if two_columns:
        axes[0,2].axvline(x=min_idx, color='red', linestyle='--', 
                         label='Divisão detectada')
        axes[0,2].legend()
    
    # Segunda linha: resultados
    axes[1,0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    axes[1,0].set_title(f"Bloco detectado\n{'2 colunas' if two_columns else '1 coluna'}")
    axes[1,0].axis("off")
    
    axes[1,1].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    axes[1,1].set_title("Bloco esquerdo")
    axes[1,1].axis("off")
    
    axes[1,2].imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
    axes[1,2].set_title("Bloco direito")
    axes[1,2].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    print(f"Bloco detectado: {w_crop}x{h_crop} pixels")
    print(f"Posição: ({x0},{y0}) a ({x1},{y1})")
    print(f"Duas colunas detectadas: {two_columns}")
    
    return left, right

# Função otimizada especificamente para cartões resposta
def detect_answer_sheet_blocks(path, skip_top_frac=0.05, skip_bottom_frac=0.05):
    """
    Versão otimizada para cartões resposta com dois blocos lado a lado.
    Ignora cabeçalho maior e foca na área dos blocos de questões.
    """
    img = cv2.imread(path)
    H, W = img.shape[:2]
    
    # Define região dos blocos (ignora mais do topo para pular cabeçalho)
    y_start = int(H * skip_top_frac)
    y_end = int(H * (1 - skip_bottom_frac))
    
    # Crop da região dos blocos
    crop = img[y_start:y_end, :]
    
    # Converte para escala de cinza e aplica threshold
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_th = cv2.adaptiveThreshold(crop_gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 8)
    
    # Projeção vertical para encontrar a separação entre blocos
    h_crop, w_crop = crop.shape[:2]
    col_projection = np.sum(crop_th > 0, axis=0)
    
    # Encontra a região central onde deve estar a separação
    center = w_crop // 2
    search_range = w_crop // 6  # busca em ±1/6 da largura ao redor do centro
    
    start_search = max(0, center - search_range)
    end_search = min(w_crop, center + search_range)
    
    # Encontra o ponto com menor densidade na região central
    min_col = start_search + np.argmin(col_projection[start_search:end_search])
    
    # Refina a divisão procurando por uma região de baixa densidade
    # ao invés de um único ponto
    window_size = 20
    best_split = min_col
    min_density = float('inf')
    
    for i in range(max(0, min_col - window_size//2), 
                   min(w_crop - window_size, min_col + window_size//2)):
        window_density = np.mean(col_projection[i:i+window_size])
        if window_density < min_density:
            min_density = window_density
            best_split = i + window_size//2
    
    # Divide os blocos
    left = crop[:, :best_split]
    right = crop[:, best_split:]
    
    return left, right, crop, col_projection, best_split

# Função alternativa mais simples focada apenas na área central
def detect_question_blocks_simple(path, skip_top_frac=0.35, skip_bottom_frac=0.05):
    """
    Versão mais simples que pega a região central da imagem,
    ignorando cabeçalho e rodapé.
    """
    img = cv2.imread(path)
    H, W = img.shape[:2]
    
    # Define região central
    y_start = int(H * skip_top_frac)
    y_end = int(H * (1 - skip_bottom_frac))
    
    # Crop da região central
    crop = img[y_start:y_end, :]
    h_crop, w_crop = crop.shape[:2]
    
    # Divide no meio
    mid = w_crop // 2
    left = crop[:, :mid]
    right = crop[:, mid:]
    
    # Visualização simples
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Região central")
    ax[0].axis("off")
    
    ax[1].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Bloco esquerdo")
    ax[1].axis("off")
    
    ax[2].imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
    ax[2].set_title("Bloco direito")
    ax[2].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    return left, right

# Teste com todas as versões
if __name__ == "__main__":
    path = 'WhatsApp Image 2025-07-07 at 14.48.06.jpeg'
    
    print("=== Versão otimizada para cartão resposta ===")
    left_opt, right_opt, crop_opt, projection, split_point = detect_answer_sheet_blocks(path)
    
    # Visualiza a versão otimizada
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Mostra a projeção vertical com o ponto de divisão
    axes[0,0].plot(projection)
    axes[0,0].axvline(x=split_point, color='red', linestyle='--', linewidth=2)
    axes[0,0].set_title('Projeção Vertical (Linha vermelha = divisão)')
    axes[0,0].set_xlabel('Posição X')
    axes[0,0].set_ylabel('Pixels preenchidos')
    
    # Mostra o crop completo com a linha de divisão
    axes[0,1].imshow(cv2.cvtColor(crop_opt, cv2.COLOR_BGR2RGB))
    axes[0,1].axvline(x=split_point, color='red', linestyle='--', linewidth=2)
    axes[0,1].set_title('Região dos blocos + divisão')
    axes[0,1].axis('off')
    
    # Mostra os blocos separados
    axes[1,0].imshow(cv2.cvtColor(left_opt, cv2.COLOR_BGR2RGB))
    axes[1,0].set_title('BLOCO 1 (Esquerdo)')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(cv2.cvtColor(right_opt, cv2.COLOR_BGR2RGB))
    axes[1,1].set_title('BLOCO 2 (Direito)')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Ponto de divisão: coluna {split_point}")
    print(f"Bloco 1: {left_opt.shape[1]}x{left_opt.shape[0]} pixels")
    print(f"Bloco 2: {right_opt.shape[1]}x{right_opt.shape[0]} pixels")
    
    print("\n=== Versão avançada ===")
    left1, right1 = detect_question_blocks(path)
    
    print("\n=== Versão simples ===")
    left2, right2 = detect_question_blocks_simple(path)