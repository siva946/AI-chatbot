import os
import logging
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv

app = Flask(__name__, template_folder='templates')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.error("GEMINI_API_KEY not found in .env file")
    raise ValueError("GEMINI_API_KEY is required")

def call_gemini_api(
    prompt_parts,       
    model_name="gemini-2.5-flash",
    generation_config=None,
    safety_settings=None
):
    try:
        if not prompt_parts or not prompt_parts[0].strip():
            raise ValueError("Empty prompt provided")
        
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(
            prompt_parts,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        if not response or not response.text:
            raise ValueError("Empty response from API")
            
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        return f"I'm sorry, I encountered an error processing your request."

@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return "Service temporarily unavailable", 500

@app.route("/generate_text", methods=["POST"])
def generate_text():
    try:
        if not request.json:
            return jsonify({"error": "Invalid JSON"}), 400
            
        user_text = request.json.get('prompt', '').strip()
        if not user_text:
            return jsonify({"error": "No prompt provided"}), 400
        
        if len(user_text) > 5000:
            return jsonify({"error": "Prompt too long"}), 400

        try:
            response_text = call_gemini_api([user_text])
            return jsonify({"response": response_text})
        except ValueError as ve:
            logger.error(f"ValueError in generate_text: {ve}")
            return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in generate_text: {e}")
        try:
            return jsonify({"error": "Internal server error"}), 500
        except Exception:
            return "Internal server error", 500

@app.route("/rewrite", methods=["POST"])
def rewrite_text():
    try:
        if not request.json:
            return jsonify({"error": "Invalid JSON"}), 400
            
        original_text = request.json.get('text', '').strip()
        tone = request.json.get('tone', 'neutral')
        
        if not original_text:
            return jsonify({"error": "No text provided for rewriting"}), 400
        
        if len(original_text) > 5000:
            return jsonify({"error": "Text too long"}), 400

        prompt = f"Rewrite the following text in a {tone} tone:\n\n{original_text}"
        response_text = call_gemini_api([prompt], model_name="gemini-1.5-pro-latest")
        return jsonify({"response": response_text})
    except Exception as e:
        logger.error(f"Error in rewrite_text: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/summarize", methods=["POST"])
def summarize_text():
    try:
        if not request.json:
            return jsonify({"error": "Invalid JSON"}), 400
            
        original_text = request.json.get('text', '').strip()
        length = request.json.get('length', 'medium')
        
        if not original_text:
            return jsonify({"error": "No text provided for summarizing"}), 400
        
        if len(original_text) > 10000:
            return jsonify({"error": "Text too long"}), 400

        prompt = f"Summarize the following text to a {length} length:\n\n{original_text}"
        response_text = call_gemini_api([prompt], model_name="gemini-1.5-pro-latest")
        return jsonify({"response": response_text})
    except Exception as e:
        logger.error(f"Error in summarize_text: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/generate_code", methods=["POST"])
def generate_code():
    try:
        if not request.json:
            return jsonify({"error": "Invalid JSON"}), 400
            
        code_prompt = request.json.get('prompt', '').strip()
        language = request.json.get('language', 'python')
        
        if not code_prompt:
            return jsonify({"error": "No code prompt provided"}), 400
        
        if len(code_prompt) > 2000:
            return jsonify({"error": "Prompt too long"}), 400

        prompt = f"Generate {language} code for: {code_prompt}\nProvide only the code block, no extra text."
        response_text = call_gemini_api([prompt], model_name="gemini-1.5-pro-latest",
                                        generation_config={"temperature": 0.5, "max_output_tokens": 800})
        return jsonify({"response": response_text})
    except Exception as e:
        logger.error(f"Error in generate_code: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/translate", methods=["POST"])
def translate_text():
    try:
        if not request.json:
            return jsonify({"error": "Invalid JSON"}), 400
            
        text_to_translate = request.json.get('text', '').strip()
        target_lang = request.json.get('target_lang', 'Tamil')
        
        if not text_to_translate:
            return jsonify({"error": "No text provided for translation"}), 400
        
        if len(text_to_translate) > 5000:
            return jsonify({"error": "Text too long"}), 400

        prompt = f"Translate the following text to {target_lang}:\n\n{text_to_translate}"
        response_text = call_gemini_api([prompt], model_name="gemini-1.5-flash-latest")
        return jsonify({"response": response_text})
    except Exception as e:
        logger.error(f"Error in translate_text: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/adjust_tone", methods=["POST"])
def adjust_tone():
    try:
        if not request.json:
            return jsonify({"error": "Invalid JSON"}), 400
            
        text = request.json.get('text', '').strip()
        tone = request.json.get('tone', '').strip()
        
        if not text or not tone:
            return jsonify({"error": "Text and tone are required"}), 400
        
        if len(text) > 5000:
            return jsonify({"error": "Text too long"}), 400

        prompt = f"Adjust the tone of the following text to be more {tone}:\n\n{text}"
        response_text = call_gemini_api([prompt], model_name="gemini-1.5-pro-latest")
        return jsonify({"response": response_text})
    except Exception as e:
        logger.error(f"Error in adjust_tone: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/generate_image_description", methods=["POST"])
def generate_image_description():
    try:
        if not request.json:
            return jsonify({"error": "Invalid JSON"}), 400
            
        prompt = request.json.get('prompt', '').strip()
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        if len(prompt) > 2000:
            return jsonify({"error": "Prompt too long"}), 400

        response_text = call_gemini_api([f"Generate a detailed image generation prompt for an AI, based on the following idea: {prompt}"])
        return jsonify({"response": response_text})
    except Exception as e:
        logger.error(f"Error in generate_image_description: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=False, host='127.0.0.1', port=5000)