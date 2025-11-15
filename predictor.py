import os
import torch
from flask import Flask, request, jsonify
from google.cloud import storage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Initialize Flask app
app = Flask(__name__)

# Global variable to hold the loaded model
llm = None


# --- Configuration ---
# These environment variables are required. The script will fail if they are not set.
GCS_BUCKET = os.getenv("GCS_BUCKET")
MODEL_NAME = os.getenv("MODEL_NAME")

if not GCS_BUCKET or not MODEL_NAME:
    raise ValueError("Error: Both GCS_BUCKET and MODEL_NAME environment variables must be set.")
# --- End Configuration ---


def download_model_files():
    """
    Downloads model files from GCS to a local directory, skipping if they already exist.
    """
    gcs_uri = f"gs://{GCS_BUCKET}/tuned-models/{MODEL_NAME}/"
    local_model_dir = f"/tmp/tuned-models/{MODEL_NAME}"

    # Check if the model directory exists and is not empty.
    if os.path.exists(local_model_dir) and os.listdir(local_model_dir):
        print(f"Model files found in {local_model_dir}. Skipping download.")
        return local_model_dir

    # If the directory doesn't exist, create it.
    if not os.path.exists(local_model_dir):
        os.makedirs(local_model_dir)

    print(f"Downloading model files from {gcs_uri} to {local_model_dir}...")
    
    # Parse the GCS URI
    bucket_name = gcs_uri.replace("gs://", "").split("/")[0]
    prefix = "/".join(gcs_uri.replace("gs://", "").split("/")[1:])

    # Download files
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        
        file_path = os.path.join(local_model_dir, os.path.basename(blob.name))
        blob.download_to_filename(file_path)
        print(f"Downloaded {blob.name} to {file_path}")
    
    print("Model files downloaded successfully.")
    return local_model_dir

def load_model():
    """
    Loads the fine-tuned model from GCS into the global llm variable.
    """
    global llm
    
    local_model_path = download_model_files()
    
    print(f"Loading model from local directory {local_model_path}...")
    
    # Diagnostic code to check GPU availability
    if torch.cuda.is_available():
        print("CUDA is available. Listing GPUs:")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    tokenizer.chat_template = None

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        dtype=torch.float16,
        device_map="auto"  # Let transformers handle device placement
    )

    print("Building text generation pipeline...")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )

    print(f"Pipeline is running on device: {pipe.device}")

    llm = HuggingFacePipeline(pipeline=pipe)
    
    print("Model loaded successfully.")

def parse_llama_output(raw_output: str) -> str:
    """
    Parses the raw output from the Llama model to extract only the assistant's response.
    """
    delimiter = "<|start_header_id|>assistant<|end_header_id|>"
    if delimiter in raw_output:
        return raw_output.split(delimiter)[-1].strip()
    return raw_output.strip()

def create_chain(template, input_variable):
    """Creates a LangChain chain with a given template and input variable."""
    prompt_template = PromptTemplate.from_template(template)
    return (
        {input_variable: RunnablePassthrough()}
        | prompt_template
        | llm
        | RunnableLambda(parse_llama_output)
    )


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles prediction requests by answering a question.
    """
    if not llm:
        return jsonify({"error": "Model is not loaded yet"}), 503

    try:
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an AI assistant that mimics the professional persona of Nico Westerdale.<|eot_id|><|start_header_id|>user<|end_header_id|>

Answer the following question in his voice: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        chain = create_chain(template, "question")
        data = request.get_json()
        prompts = [instance['prompt'] for instance in data['instances']]
        results = chain.batch(prompts)
        
        return jsonify({"predictions": results})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400


@app.route('/rewrite', methods=['POST'])
def rewrite():
    """
    Handles rewrite requests by translating text into Nico's voice.
    """
    if not llm:
        return jsonify({"error": "Model is not loaded yet"}), 503

    try:
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You rewrite text in your own voice.<|eot_id|><|start_header_id|>user<|end_header_id|>

Rewrite the following text in your voice: {original_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        chain = create_chain(template, "original_text")
        data = request.get_json()
        prompts = [instance['original_text'] for instance in data['instances']]
        results = chain.batch(prompts)
        
        return jsonify({"predictions": results})
    except Exception as e:
        print(f"Error during rewrite: {e}")
        return jsonify({"error": str(e)}), 400


@app.route('/prompt', methods=['POST'])
def prompt():
    """
    Handles generic prompt requests.
    Expects a JSON body with:
    - "system_prompt": "You are a helpful assistant."
    - "user_prompt": "Translate the following to French: {text}"
    - "instances": [{"text": "Hello, world!"}]
    The user_prompt must contain the placeholder {text}.
    """
    if not llm:
        return jsonify({"error": "Model is not loaded yet"}), 503

    try:
        data = request.get_json()
        
        system_prompt = data.get('system_prompt', '')
        user_prompt = data['user_prompt']
        
        if '{text}' not in user_prompt:
            return jsonify({"error": "user_prompt must contain a {text} placeholder."}), 400

        if system_prompt:
            template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            template = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        chain = create_chain(template, "text")
        
        if 'instances' not in data or not isinstance(data['instances'], list):
             return jsonify({"error": "Request body must contain a list of 'instances'."}), 400

        # Each instance is expected to be a dict like {"text": "..."}
        prompts = [instance['text'] for instance in data['instances']]
        
        results = chain.batch(prompts)
        
        return jsonify({"predictions": results})
    except KeyError as e:
        return jsonify({"error": f"Missing required key in request body: {e}"}), 400
    except Exception as e:
        print(f"Error during prompt: {e}")
        return jsonify({"error": str(e)}), 400


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for Cloud Run.
    Returns OK if the model is loaded.
    """
    if llm:
        return "OK", 200
    else:
        return "Model not loaded", 503

# Load the model when the application starts
print("Hi Nico, not ready yet, let me get my model...")
print("Starting model load...")
load_model()
print("Model loaded successfully.")

if __name__ == '__main__':
    print("Starting Flask server...")
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
