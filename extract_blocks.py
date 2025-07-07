import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) Carrega a imagem e faz um resize “suave” (opcional, só se a imagem for enorme)
image_path = 'WhatsApp Image 2025-07-04 at 14.04.06 (1).jpeg'
orig = cv2.imread(image_path)
img = orig.copy()
h_img, w_img = img.shape[:2]

# 2) Converte pra tons de cinza e aplica Adaptive Threshold (invertido)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
th = cv2.adaptiveThreshold(
    gray, 
    maxValue=255, 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV, 
    blockSize=15, 
    C=10
)

# (Opcional) Mostrando o resultado do threshold pra referência:
plt.figure(figsize=(6,6))
plt.title("Adaptive Threshold")
plt.imshow(th, cmap='gray')
plt.axis('off')

# 3) Encontra TODOS os contornos (RETR_EXTERNAL para pegar só o contorno externo de cada “mancha” branca)
contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# 4) Dentre todos os contornos, escolhe aquele que engloba OS 4 blocos, mas não é a borda inteira da folha.
#    Para isso, filtramos pelo contorno cuja bounding box:
#      - Tenha área entre 10% e 90% da área total da imagem.
#      - Não seja exatamente a folha inteira (w < w_img e h < h_img).
img_area = w_img * h_img
candidatos = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 0.10 * img_area and area < 0.90 * img_area:
        x, y, w, h = cv2.boundingRect(cnt)
        # descartamos retângulos que ocupem 100% da largura ou altura
        if w < w_img and h < h_img:
            candidatos.append((area, cnt))

if len(candidatos) == 0:
    raise RuntimeError("Não encontrou nenhum contorno grande como bloco de respostas.")

# 5) Pega o contorno de maior área entre esses candidatos
maior_cnt = max(candidatos, key=lambda x: x[0])[1]
x_bloco, y_bloco, w_bloco, h_bloco = cv2.boundingRect(maior_cnt)

# Desenha o retângulo encontrado (em verde) só pra checar:
vis = orig.copy()
cv2.rectangle(vis, (x_bloco, y_bloco), (x_bloco + w_bloco, y_bloco + h_bloco), (0, 255, 0), 3)
plt.figure(figsize=(8,6))
plt.title("Contorno Único dos 4 blocos")
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.axis('off')

print(f"Contorno principal localizado em: x={x_bloco}, y={y_bloco}, w={w_bloco}, h={h_bloco}")

# 6) “Corta” (crop) essa região: aqui temos todos os 4 blocos alinhados
crop_all_blocks = img[y_bloco : y_bloco + h_bloco, x_bloco : x_bloco + w_bloco]

# Se quiser ver o recorte:
plt.figure(figsize=(6,6))
plt.title("Crop com os 4 blocos")
plt.imshow(cv2.cvtColor(crop_all_blocks, cv2.COLOR_BGR2RGB))
plt.axis('off')

# 7) Agora precisamos separar em 4 blocos individuais. 
#    Observando a foto, os 4 blocos estão lado a lado e têm larguras muito parecidas.
#    Vamos fatiar em 4 colunas iguais (em px):
w_crop, h_crop = w_bloco, h_bloco
largura_por_bloco = w_crop // 4

# Guardamos num array para processar depois (ou mostrar separadamente)
blocks = []
for i in range(4):
    x0 = i * largura_por_bloco
    # para o último bloco, vai até o final exato (para englobar eventuais pixels extras)
    x1 = (i + 1) * largura_por_bloco if i < 3 else w_crop
    bloco_i = crop_all_blocks[:, x0:x1]
    blocks.append(bloco_i)

    # Para efeitos de visualização, podemos desenhar linhas verticais (azuis) na imagem original:
    xv = x_bloco + x0
    cv2.line(vis, (xv, y_bloco), (xv, y_bloco + h_bloco), (255, 0, 0), 2)

# Desenha também a borda à direita do último bloco
xv_final = x_bloco + w_crop
cv2.line(vis, (xv_final, y_bloco), (xv_final, y_bloco + h_bloco), (255, 0, 0), 2)

# Mostrar a divisão em 4 colunas:
plt.figure(figsize=(8, 6))
plt.title("Divisão dos 4 blocos (linhas azuis) e contorno geral (verde)")
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.axis('off')

# 8) Se quiser salvar cada bloco individual em disco:
for idx, b in enumerate(blocks, start=1):
    filename = f"bloco_{idx}.png"
    cv2.imwrite(filename, b)
    print(f"Bloco {idx} salvo em: {filename}")

# 9) Exibir os 4 blocos separadamente (opcional)
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for i, b in enumerate(blocks):
    axs[i].imshow(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
    axs[i].set_title(f"Bloco {i+1}")
    axs[i].axis('off')
plt.show()
