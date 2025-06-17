from flask import Flask, request, jsonify
import tempfile
import os
import asyncio
from werkzeug.utils import secure_filename

from gemini_model import( 
    segment_image_blocks,
    process_blocks_async
)

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze_image():
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
            try:
                result.append({"block": i, "response": r})
            except Exception as e:
                result.append({"block": i, "error": str(e)})

        return {
            "status": "success",
            "blocks": result
        }

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Remove o arquivo temporário
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
