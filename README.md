### LangChain-Huggingface-Transformers-Integration

This repository provides a **comprehensive guide for integrating LangChain and Hugging Face Transformer models in Python**. It covers virtual environment setup, dependency installation, and **CPU-only execution** (or optional GPU acceleration with CUDA/PyTorch) [Previous Turn]. You will learn to run models like **Mistral-7B-Instruct-v0.1 efficiently**, ensuring a seamless setup to leverage both frameworks.

This guide walks you through setting up a Python environment, installing dependencies, configuring your PyTorch installation for CPU (or GPU), and running a transformer model with LangChain.

#### 1. Create a Virtual Environment

Creating a virtual environment helps isolate dependencies and prevents conflicts with other Python projects.

*   **For Windows (Command Prompt)**
    ```bash
    python -m venv langchain-env
    langchain-env\Scripts\activate
    ```
*   **For macOS/Linux (Terminal)**
    ```bash
    python -m venv langchain-env
    source langchain-env/bin/activate
    ```

#### 2. Install Requirements

Once the virtual environment is activated, install the required dependencies.

You will need `langchain`, `transformers`, and `langchain-huggingface`.
```bash
pip install langchain transformers langchain-huggingface
```

#### 3. Install PyTorch (CPU-Only or GPU)

To run the models, you need PyTorch.

*   **For CPU-Only Execution:**
    If you are not using a GPU, install PyTorch with CPU support using the following command [Previous Turn]:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

*   **For GPU Acceleration (Optional):**
    If you have an NVIDIA GPU and wish to leverage it, install the CUDA-enabled version of PyTorch. Remember to **replace `cu126` with your specific CUDA version**:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```
    To check which CUDA version you have installed, run:
    ```bash
    nvcc --version
    ```
    If you don’t have CUDA installed, you can follow the official installation guide: 🔗 CUDA Installation Guide.

#### 4. Check for PyTorch Device Availability

It's good practice to verify your PyTorch installation. If you installed the CPU-only version, `torch.cuda.is_available()` will return `False`.

Run the following Python code to verify:
```python
import torch

# Check if GPU is available (will be False if CPU-only PyTorch is installed)
gpu_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if gpu_available else "No GPU found"

print(f"GPU Available: {gpu_available}")
print(f"GPU Name: {device_name}")
print(f"Using device: {'cuda' if gpu_available else 'cpu'}")
```
If `torch.cuda.is_available()` returns `False` when you expected GPU support, ensure that:
*   You have an NVIDIA GPU.
*   The correct version of CUDA is installed.
*   You installed the CUDA-enabled version of PyTorch.

#### 5. Set Device in Pipeline

Once your PyTorch installation is confirmed, specify the device in the transformer pipeline to match your setup. This repository specifically focuses on running models like **Mistral-7B-Instruct-v0.1**.

**For CPU-only execution (as per your setup), you must set `device="cpu"`** [17, Previous Turn]:
```python
from transformers import pipeline

# Load the model and set device to CPU
model = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    device="cpu"  # Use CPU
)

# Generate text
output = model("What is LangChain?")
print(output)
```
If you are using a GPU, change `device="cpu"` to `device=0` (or the appropriate GPU index).

#### 6. Hugging Face Account & Token (Recommended)

Many Hugging Face models require that you accept their terms or license agreement before you are able to download and use them. It is recommended to create an account on Hugging Face and obtain an access token.

To log in via the command line after installing `transformers`:
```bash
huggingface-cli login
```
You will be prompted to paste your token.

#### 7. Finding Hugging Face Models

You can access thousands of free AI models from Hugging Face locally on your own computer using the `transformers` package. To find models that work with the `transformers` library, you can go to the Hugging Face website, click on "Models", then filter by "Libraries" and select "Transformers". You can then search and filter by tasks such as "text summarization," "text classification," or "text generation".

Each model's page will typically provide an example of exactly how to use it from your code and its documentation.

#### 🎯 Summary

Here’s a quick summary of the key steps for your reference:

| Step                       | Command / Code                                                              |
| :------------------------- | :-------------------------------------------------------------------------- |
| **Create a Virtual Env**   | `python -m venv langchain-env && source langchain-env/bin/activate`         |
| **Install Requirements**   | `pip install langchain transformers langchain-huggingface`                  |
| **Install PyTorch (CPU)**  | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` |
| **Install PyTorch (GPU)**  | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126` |
| **Check Device Availability** | `print(torch.cuda.is_available())`                                          |
| **Run Model on CPU**       | `pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device=-1)` |
| **Run Model on GPU**       | `pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device=0)` |
| **Hugging Face Login**     | `huggingface-cli login`                                                |

Now you’re ready to use **LangChain and Hugging Face Transformer models with CPU (or GPU) support!** 