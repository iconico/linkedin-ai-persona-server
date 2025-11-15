# LinkedinIn AI Persona Server

A Gunicorn + Flask server designed to host a fine-tuned Large Language Model (LLM) and serve it via a RESTful API. It is built for a "scale-to-zero" deployment on serverless platforms, specifically Google Cloud Run.

On startup, this server downloads the multi-gigabyte model artifacts from a private Google Cloud Storage bucket into a temporary directory before loading it into memory (preferably VRAM). It then exposes a set of RESTful API endpoints for interaction.

## API Endpoints

All endpoints expect a `POST` request with a JSON body. The first request to a new or scaled-down instance will be agonizingly slow due to the "cold start" process of downloading and loading the model. Subsequent requests will be faster.

### 1. `/predict`

Answers a direct question in the target persona's voice.

**Request Body:**

```json
{
  "instances": [
    { "prompt": "Your question here" }
  ]
}
```

**Example (PowerShell):**

```powershell
Invoke-RestMethod -Uri "YOUR_CLOUD_RUN_URL/predict" -Method Post -ContentType "application/json" -Body '{"instances": [{"prompt": "What is the key to building a successful tech startup?"}]}' -TimeoutSec 3600
```

### 2. `/rewrite`

Translates a given piece of text into the target persona's voice.

**Request Body:**

```json
{
  "instances": [
    { "original_text": "The text you want to be rewritten" }
  ]
}
```

**Example (PowerShell):**

```powershell
Invoke-RestMethod -Uri "YOUR_CLOUD_RUN_URL/rewrite" -Method Post -ContentType "application/json" -Body '{"instances": [{"original_text": "Our team is focused on leveraging synergies to maximize stakeholder value."}]}' -TimeoutSec 3600
```

### 3. `/prompt`

A generic, flexible endpoint that allows for custom system and user prompts. This is for advanced prompt engineering. The `user_prompt` *must* contain a `{text}` placeholder.

**Request Body:**

```json
{
  "system_prompt": "Your custom system prompt.",
  "user_prompt": "Your custom user prompt with a {text} placeholder.",
  "instances": [
    { "text": "The text to be inserted into the placeholder." }
  ]
}
```

**Example (PowerShell):**

```powershell
Invoke-RestMethod -Uri "YOUR_CLOUD_RUN_URL/prompt" -Method Post -ContentType "application/json" -Body '{"system_prompt": "You are a jaded, cynical venture capitalist.", "user_prompt": "Destroy this startup idea in one sentence: {text}", "instances": [{"text": "It''s like Uber, but for pet rocks."}]}' -TimeoutSec 3600
```

## Local Development & GPU Hell

Running this server locally is possible, but requires a specific setup, a powerful GPU, and a great deal of patience.

**Prerequisites:**

*   Python 3.11.9 (do not use a newer version, or PyTorch will fail).
*   An NVIDIA GPU with at least 12GB of VRAM.
*   The `gcloud` CLI, authenticated to the correct Google Cloud project.

**Setup Steps:**

1.  **Configure Environment:** Create a `.env` file in this directory by copying the `.env.sample`. Fill in the required values:
    *   `GCS_BUCKET`: The name of the Google Cloud Storage bucket where your models are stored.
    *   `MODEL_NAME`: The specific model folder you want the server to load.

2.  **Create a Virtual Environment:** From this directory, create and activate a Python virtual environment.
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

3.  **Install Dependencies:** Install all the necessary Python packages. This will take some time.
    ```powershell
    pip install -r requirements.txt
    ```

4.  **Authenticate for GCS:** The server needs to download the model from GCS. You must provide Application Default Credentials.
    ```powershell
    gcloud auth application-default login
    ```
    Follow the prompts in your browser and ensure you are authenticating with an account that has at least "Storage Object Viewer" permissions on the GCS bucket where your model is stored.

5.  **Run the Server:** Start the Flask development server.
    ```powershell
    python predictor.py
    ```
    The console will hang for several minutes after printing "Starting model load...". It is downloading the model files from the GCS bucket specified in your `.env` file and then loading them onto your GPU. Do not interrupt it.

6.  **Verify GPU Usage (Optional but Recommended):** To ensure your expensive hardware is actually being used, open a separate terminal and run `nvidia-smi` to monitor GPU VRAM and utilization during model load and inference.

## Deployment to Google Cloud Run

This server is designed for a "scale-to-zero" deployment on Google Cloud Run.

1.  **Build & Push the Docker Image:** From this directory, run the following command. This builds the Docker image and pushes it to your project's Google Container Registry. Remember to replace `YOUR_GCP_PROJECT_ID` with your actual project ID.
    ```powershell
    gcloud builds submit --region=us-central1 --tag gcr.io/YOUR_GCP_PROJECT_ID/ai-persona-server:latest .
    ```

2.  **Deploy New Revision:** In the Google Cloud Console, navigate to your Cloud Run service and choose to "Edit and Deploy New Revision". Configure it with the following **critical** settings:
    *   **Image:** Select the `:latest` tag you just pushed.
    *   **Variables & Secrets Tab:** Add the `GCS_BUCKET` and `MODEL_NAME` environment variables and set them to the appropriate values for your production model.
    *   **CPU Allocation:** Set to **"CPU is only allocated during request processing"**. This is the key to being cheap.
    *   **Resources:** 8 vCPU, 32 GiB Memory.
    *   **Healthchecks (Startup Probe):**
        *   **Request Path:** `/health`
        *   **Timeout:** `600` seconds (to allow time for the model download and load).
        *   **Number of retries:** `5`
