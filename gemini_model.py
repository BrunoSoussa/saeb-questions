import base64
import os
from google import genai
from google.genai import types
import cv2
import asyncio
import json
import dotenv

dotenv.load_dotenv()

# Importa as funções de segmentação (adapte o nome do arquivo conforme necessário)
from separed import detect_answer_sheet_blocks, detect_question_blocks_simple, detect_question_blocks

def load_image_as_base64(image_bytes):
    """Carrega bytes de imagem e retorna a representação em base64."""
    return base64.b64encode(image_bytes).decode("utf-8")

def segment_image_blocks(image_path, method='optimized'):
    """
    Segmenta a imagem em blocos usando uma das funções disponíveis.
    
    Args:
        image_path: caminho para a imagem
        method: 'optimized', 'advanced', ou 'simple'
    
    Returns:
        list: lista com os blocos [left_block, right_block]
    """
    try:
        if method == 'optimized':
            # Usa a versão otimizada para cartões resposta
            left, right, crop, projection, split_point = detect_answer_sheet_blocks(image_path)
            print(f"Método otimizado: divisão na coluna {split_point}")
            
        elif method == 'advanced':
            # Usa a versão avançada
            left, right = detect_question_blocks(image_path)
            print("Método avançado usado")
            
        elif method == 'simple':
            # Usa a versão simples
            left, right = detect_question_blocks_simple(image_path)
            print("Método simples usado")
            
        else:
            raise ValueError("Método deve ser 'optimized', 'advanced' ou 'simple'")
        
        if left is None or right is None:
            print("Erro: Não foi possível segmentar os blocos")
            return None
            
        # Verifica se os blocos têm tamanho válido
        if left.size == 0 or right.size == 0:
            print("Erro: Um dos blocos está vazio")
            return None
            
        print(f"Bloco esquerdo: {left.shape}")
        print(f"Bloco direito: {right.shape}")
        
        return [left, right]
        
    except Exception as e:
        print(f"Erro na segmentação: {e}")
        return None

async def analyze_block_async(image_block, block_id):
    """Analisa um único bloco de imagem com a API Gemini de forma assíncrona usando schema."""
    try:
        client = genai.Client(
            api_key=os.getenv("GENAI_API_KEY"),
        )

        model = "gemini-2.0-flash"

        # Verifica se o bloco é válido
        if image_block is None or image_block.size == 0:
            print(f"Bloco {block_id} inválido")
            return None

        # Codifica imagem em PNG e converte para base64
        _, img_encoded = cv2.imencode('.png', image_block)
        base64_image = load_image_as_base64(img_encoded.tobytes())

        # Define o schema esperado para a resposta
        response_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "block_id": types.Schema(type=types.Type.INTEGER),
                "has_unmarked": types.Schema(type=types.Type.BOOLEAN),
                "has_duplicated": types.Schema(type=types.Type.BOOLEAN),
                "questions_marked_processed": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                ),
                "is_valid_img": types.Schema(type=types.Type.BOOLEAN),
            },
            required=[
                "block_id",
                "has_unmarked",
                "has_duplicated",
                "questions_marked_processed",
                "is_valid_img",
            ],
        )

        # Calcula o range de questões para cada bloco
        start_question = (block_id - 1) * 11 + 1  # Assumindo 11 questões por bloco
        end_question = block_id * 11

        contents = [
            types.Content(
                role="user",
                parts=[
                    # Anexo da imagem
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/png",
                            data=base64.b64decode(base64_image),
                        )
                    ),
                    # Prompt solicitando a análise
                    types.Part.from_text(
                        text=(
                            f"Analise a imagem contendo um bloco de questões de múltipla escolha "
                            f"(questões de {start_question} a {end_question}, alternativas A, B, C, D). "
                            f"Para cada questão, identifique qual alternativa está marcada/pintada. "
                            f"Retorne um JSON seguindo o schema configurado onde:\n"
                            f"- block_id: {block_id}\n"
                            f"- has_unmarked: true se alguma questão não tem nenhuma alternativa marcada\n"
                            f"- has_duplicated: true se alguma questão tem múltiplas alternativas marcadas\n"
                            f"- questions_marked_processed: array com as respostas no formato ['1':'A', '2':'B', '3':'C', etc.]\n"
                            f"- is_valid_img: true se a imagem é legível e processável\n"
                            f"Considere apenas marcações claras e visíveis (círculos pintados/preenchidos)."

                            ''' exemplo de resposta:
                            {
                                "block_id": 1,
                                "has_unmarked": false,
                                "has_duplicated": false,
                                "questions_marked_processed": [
                                    "1:A",
                                    "2:B",
                                    "3:C",
                                    "4:D",
                                    "5:A",
                                    "6:B",
                                    "7:C",
                                    "8:D",
                                    "9:A",
                                    "10:B",
                                    "11:C"
                                ],
                                "is_valid_img": true
                            }

                            '''
                        )
                    ),
                ],
            )
        ]

        # Configura a geração para usar o schema
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0
        )

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        try:
            # Tenta decodificar a resposta JSON
            response_json = json.loads(response.text)
            return response_json
        except json.JSONDecodeError:
            print(f"Erro ao decodificar JSON da resposta do bloco {block_id}")
            return None
        
        
        
    except Exception as e:
        print(f"Erro ao analisar bloco {block_id}: {e}")
        return None

async def process_blocks_async(segmented_blocks):
    """Processa todos os blocos de imagem de forma assíncrona."""
    if not segmented_blocks:
        print("Nenhum bloco para processar")
        return []
        
    tasks = [
        analyze_block_async(block, idx)
        for idx, block in enumerate(segmented_blocks, start=1)
    ]
    return await asyncio.gather(*tasks)

def save_blocks_for_debug(segmented_blocks, prefix="debug_block"):
    """Salva os blocos como imagens para debug."""
    if not segmented_blocks:
        return
        
    for i, block in enumerate(segmented_blocks, start=1):
        if block is not None and block.size > 0:
            filename = f"{prefix}_{i}.png"
            cv2.imwrite(filename, block)
            print(f"Bloco {i} salvo como {filename}")

def main():
    """Função principal que processa a imagem e analisa os blocos."""
    image_path = "WhatsApp Image 2025-06-17 at 14.50.53.jpeg"  # Ajuste o nome do arquivo
    
    print("=== Iniciando Segmentação ===")
    
    # Testa diferentes métodos de segmentação
    methods = ['optimized', 'simple', 'advanced']
    
    for method in methods:
        print(f"\n--- Testando método: {method} ---")
        segmented_blocks = segment_image_blocks(image_path, method=method)
        
        if segmented_blocks:
            print(f"Segmentação bem-sucedida com método '{method}'")
            
            # Salva blocos para debug (opcional)
            save_blocks_for_debug(segmented_blocks, f"debug_{method}")
            
            # Processa com Gemini
            print("Iniciando análise assíncrona dos blocos...")
            try:
                responses = asyncio.run(process_blocks_async(segmented_blocks))
                print("--- Resultados da Análise ---")
                for i, resp in enumerate(responses, start=1):
                    if resp:
                        print(f"Bloco {i} resultado: {resp}\n")
                    else:
                        print(f"Bloco {i}: Erro na análise\n")
                        
                # Se chegou até aqui com sucesso, para o loop
                break
                
            except Exception as e:
                print(f"Erro na análise com Gemini: {e}")
                continue
        else:
            print(f"Falha na segmentação com método '{method}'")
    
    print("Processamento concluído!")

if __name__ == "__main__":
    main()