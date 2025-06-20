from flask import Flask, request, jsonify
import tempfile
import os
import asyncio
from werkzeug.utils import secure_filename
import dotenv

dotenv.load_dotenv()

from gemini_model import (
    segment_image_blocks,
    process_blocks_async
)

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
app.config['RATE_LIMIT'] = '100 per second'  


@app.route("/")
def index():
    return jsonify({"message": "API de Análise de Imagens"}), 200

@app.route("/analyze", methods=["POST"])
def analyze_image():
    if 'X-API-KEY' not in request.headers:
        return jsonify({"error": "Chave de API não fornecida."}), 401
    elif request.headers['X-API-KEY'] != os.getenv("X-API-KEY"):
        print("Chave de API inválida:", request.headers['X-API-KEY'])
        print("Chave de API correta:", os.getenv("X-API-KEY"))
        return jsonify({"error": "Chave de API inválida."}), 401
    
    if 'image' not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada."}), 400

    image_file = request.files['image']
    # Salva temporariamente a imagem
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg") as tmp:
        image_path = tmp.name
        image_file.save(image_path)

    try:
        # Segmenta os blocos da imagem
        segmented_blocks = segment_image_blocks(image_path, method="optimized")
        if not segmented_blocks:
            return jsonify({"error": "Falha na segmentação da imagem."}), 422

        # Analisa os blocos de forma assíncrona
        responses = asyncio.run(process_blocks_async(segmented_blocks))

        result = []
        for i, r in enumerate(responses, start=1):
            if not r or "questions_marked_processed" not in r:
                result.append({"block": i, "error": "Resposta inválida"})
                continue

            # Normaliza os itens
            normalized = {
                "questions_marked_processed": [],
                "is_valid_img": r.get("is_valid_img", False)
            }

            for item in r["questions_marked_processed"]:
                q_raw = item.get("question", "")
                a_raw = item.get("answer", None)

                # Converte a chave para inteiro (removendo zeros à esquerda)
                try:
                    q = int(q_raw.lstrip("0") or "0")
                except ValueError:
                    q = q_raw  # mantém string original se falhar

                # Converte resposta para lowercase, se for string
                if isinstance(a_raw, str):
                    a = a_raw.lower()
                else:
                    a = a_raw

                normalized["questions_marked_processed"].append({
                    "question": q,
                    "answer": a
                })

            result.append({"block": i, "response": normalized})

        return jsonify({
            "status": "success",
            "blocks": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Remove o arquivo temporário
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
