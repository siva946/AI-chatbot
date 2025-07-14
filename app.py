import os
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv

app = Flask(__name__, template_folder='templates')

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found in .env file. API calls may fail.")

def call_gemini_api(
    prompt_parts,       
    model_name="gemini-2.5-flash",
    generation_config=None,
    safety_settings=None
):
    try:
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(
            prompt_parts,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")

        if hasattr(e, 'response') and e.response:
            print(f"Gemini Response Error Details: {e.response.text}")
        return f"Error: {str(e)}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate_text", methods=["POST"])
def generate_text():
    user_text = request.json.get('prompt')
    if not user_text:
        return jsonify({"error": "No prompt provided"}), 400

    response_text = call_gemini_api([user_text])
    return jsonify({"response": response_text})

@app.route("/rewrite", methods=["POST"])
def rewrite_text():
    original_text = request.json.get('text')
    tone = request.json.get('tone', 'neutral')
    if not original_text:
        return jsonify({"error": "No text provided for rewriting"}), 400

    prompt = f"Rewrite the following text in a {tone} tone:\n\n{original_text}"
    response_text = call_gemini_api([prompt], model_name="gemini-1.5-pro-latest")
    return jsonify({"response": response_text})

@app.route("/summarize", methods=["POST"])
def summarize_text():
    original_text = request.json.get('text')
    length = request.json.get('length', 'medium')
    if not original_text:
        return jsonify({"error": "No text provided for summarizing"}), 400

    prompt = f"Summarize the following text to a {length} length:\n\n{original_text}"
    response_text = call_gemini_api([prompt], model_name="gemini-1.5-pro-latest")
    return jsonify({"response": response_text})

@app.route("/generate_code", methods=["POST"])
def generate_code():
    code_prompt = request.json.get('prompt')
    language = request.json.get('language', 'python')
    if not code_prompt:
        return jsonify({"error": "No code prompt provided"}), 400

    prompt = f"Generate {language} code for: {code_prompt}\nProvide only the code block, no extra text."

    response_text = call_gemini_api([prompt], model_name="gemini-1.5-pro-latest",
                                    generation_config={"temperature": 0.5, "max_output_tokens": 800})
    return jsonify({"response": response_text})

@app.route("/translate", methods=["POST"])
def translate_text():
    text_to_translate = request.json.get('text')
    target_lang = request.json.get('target_lang', 'Tamil')
    if not text_to_translate:
        return jsonify({"error": "No text provided for translation"}), 400

    prompt = f"Translate the following text to {target_lang}:\n\n{text_to_translate}"
    response_text = call_gemini_api([prompt], model_name="gemini-1.5-flash-latest")
    return jsonify({"response": response_text})

@app.route("/adjust_tone", methods=["POST"])
def adjust_tone():
    text = request.json.get('text')
    tone = request.json.get('tone')
    if not text or not tone:
        return jsonify({"error": "Text and tone are required"}), 400

    prompt = f"Adjust the tone of the following text to be more {tone}:\n\n{text}"
    response_text = call_gemini_api([prompt], model_name="gemini-1.5-pro-latest")
    return jsonify({"response": response_text})


@app.route("/generate_image_description", methods=["POST"])
def generate_image_description():
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400


    response_text = call_gemini_api([f"Generate a detailed image generation prompt for an AI, based on the following idea: {prompt}"])
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True)