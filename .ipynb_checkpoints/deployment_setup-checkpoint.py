import os
import sys
import subprocess
from pathlib import Path
import inquirer
import logging
import importlib.util

# ----------------------------
# Setup Logging
# ----------------------------
logging.basicConfig(filename='deployment_setup.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)

# ----------------------------
# Utility Functions
# ----------------------------

def validate_python_installation():
    """Ensure Python 3 is installed and accessible."""
    try:
        subprocess.check_call(["python3", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        logging.error("Python3 is not installed or not in PATH. Please install Python3 and rerun this script.")
        sys.exit(1)

def check_virtualenv():
    """Check if the script is running inside a virtual environment."""
    if sys.prefix == sys.base_prefix:
        logging.error("This script must be run within an active virtual environment.")
        print("Error: This script must be run within an active virtual environment.")
        sys.exit(1)
    else:
        logging.info("Running inside a virtual environment.")

def check_dependencies():
    """Check if all required dependencies are installed in the virtual environment."""
    required_modules = ["torch", "transformers", "flask", "streamlit"]
    missing_modules = []

    for module in required_modules:
        if importlib.util.find_spec(module) is None:
            missing_modules.append(module)

    if missing_modules:
        logging.error(f"Missing required dependencies: {', '.join(missing_modules)}")
        print(f"Error: The following required dependencies are missing: {', '.join(missing_modules)}")
        print("Please install them in your virtual environment using the following command:")
        print(f"   pip install {' '.join(missing_modules)}")
        sys.exit(1)

# ----------------------------
# Interactive Setup
# ----------------------------

def setup_interactive():
    """
    Guide the user through interactive setup to gather preferences for deployment.
    """
    logging.info("Starting interactive setup for Falcon 7B Deployment Tool.")
    print("""
    ===========================================
    |      Falcon 7B Deployment Setup Tool     |
    |-----------------------------------------|
    | This script will guide you through the  |
    | setup of Falcon 7B deployment, including|
    | environment setup, model selection, and |
    | deployment options.                     |
    ===========================================

    Please ensure that you are running this script within an active virtual environment.
    You should have the following dependencies installed in your virtual environment:

    Required Packages:
    - torch
    - transformers
    - flask
    - streamlit

    To create and activate a virtual environment, you can use the following commands:
        python3 -m venv venv_name
        source venv_name/bin/activate

    To install the required dependencies, run:
        pip install torch transformers flask streamlit
    """)

    validate_python_installation()
    check_virtualenv()
    check_dependencies()

    # Model licensing information
    model_info = {
        "Falcon 7B Instruct": {
            "name": "tiiuae/falcon-7b-instruct",
            "license": "Apache 2.0",
            "notes": "No special access required."
        },
        "LLaMA": {
            "name": "meta-llama/Llama-2-7b-hf",
            "license": "Custom (Meta AI License Agreement)",
            "notes": "Requires acceptance of Meta's license."
        },
        "GPT-J": {
            "name": "EleutherAI/gpt-j-6B",
            "license": "Apache 2.0",
            "notes": "No special access required."
        }
    }

    # Display model options with licensing info
    logging.info("Displaying model options for selection.")
    print("Available models:")
    for idx, (model_name, info) in enumerate(model_info.items(), start=1):
        print(f"{idx}. {model_name} - License: {info['license']} - {info['notes']}")

    current_path = str(Path.cwd())

    questions = [
        inquirer.Text('venv_path', message="Enter the path for the virtual environment (it should be already active)", default=current_path),
        inquirer.Text('deploy_path', message="Where would you like to deploy the system?", default=current_path),
        inquirer.List('model_choice',
                      message="Select the model",
                      choices=list(model_info.keys()),
                      default="Falcon 7B Instruct"),
        inquirer.Confirm('run_flask', message="Do you want a Flask-based API?", default=True),
        inquirer.Confirm('run_streamlit', message="Do you want a Streamlit frontend?", default=True),
        inquirer.Text('flask_port', message="Enter the Flask API port", default="5000")
    ]

    answers = inquirer.prompt(questions)

    venv_path = answers['venv_path']
    deploy_path = answers['deploy_path']
    model_choice = answers['model_choice']
    model_name = model_info[model_choice]['name']

    run_flask = answers['run_flask']
    run_streamlit = answers['run_streamlit']
    flask_port = answers['flask_port']

    logging.info("User selections: Virtual Environment Path: %s, Deployment Path: %s, Model: %s, Flask API: %s, Streamlit Frontend: %s", venv_path, deploy_path, model_name, run_flask, run_streamlit)

    confirm = inquirer.prompt([
        inquirer.Confirm('confirm', message="Is this information correct?", default=True)
    ])
    if not confirm['confirm']:
        logging.info("Setup cancelled by user.")
        print("Setup cancelled. Please rerun the script.")
        sys.exit(0)

    return deploy_path, model_name, run_flask, run_streamlit, flask_port, venv_path

# ----------------------------
# Script Generators
# ----------------------------

def generate_flask_script(deploy_path, model_name, flask_port):
    """
    Generate a Flask API script.
    """
    script_path = os.path.join(deploy_path, "app.py")
    os.makedirs(deploy_path, exist_ok=True)

    with open(script_path, "w") as script_file:
        script_file.write(f"""
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ----------------------------
# Configuration
# ----------------------------
MODEL_NAME = "{model_name}"

# ----------------------------
# Flask API
# ----------------------------
app = Flask(__name__)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = model.eval()

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({{"error": "No prompt provided."}}), 400
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({{"response": response}})
    except Exception as e:
        return jsonify({{"error": str(e)}}), 500

if __name__ == "__main__":
    app.run(port={flask_port})
""")
    os.chmod(script_path, 0o755)
    logging.info(f"Flask API script created at {script_path}")
    return script_path


def generate_streamlit_script(deploy_path, flask_port):
    """
    Generate a Streamlit frontend script.
    """
    script_path = os.path.join(deploy_path, "streamlit_app.py")
    with open(script_path, "w") as script_file:
        script_file.write(f"""#!/usr/bin/env python3
import streamlit as st
import requests

st.title("Language Model Interactive Interface")
prompt = st.text_area("Enter your prompt:")
if st.button("Generate"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating response..."):
            try:
                response = requests.post(f"http://localhost:{flask_port}/generate", json={{"prompt": prompt}})
                response.raise_for_status()
                st.success("Response generated:")
                st.write(response.json().get("response"))
            except requests.exceptions.ConnectionError:
                st.error("Error: Unable to connect to the API. Please ensure the Flask API is running.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {{e}}")
""")
    os.chmod(script_path, 0o755)
    logging.info(f"Streamlit script created at {script_path}")
    return script_path

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    deploy_path, model_name, run_flask, run_streamlit, flask_port, venv_path = setup_interactive()

    # Generate Flask API script
    if run_flask:
        flask_script = generate_flask_script(deploy_path, model_name, flask_port)
    else:
        flask_script = None

    # Generate Streamlit script
    if run_streamlit:
        streamlit_script = generate_streamlit_script(deploy_path, flask_port)
    else:
        streamlit_script = None

    # Prompt to run the scripts
    run_now = inquirer.prompt([
        inquirer.Confirm('run_now', message="Do you want to run the applications now?", default=True)
    ])
    if run_now['run_now']:
        logging.info("User chose to run the applications immediately.")
        try:
            if flask_script:
                subprocess.Popen(['python3', flask_script])
            if streamlit_script:
                subprocess.Popen(['streamlit', 'run', streamlit_script])
        except FileNotFoundError as e:
            logging.error(f"Failed to run the application: {e}")
            print(f"Error: {e}. Make sure Flask and Streamlit are installed in your virtual environment.")
    else:
        logging.info("User chose not to run the applications immediately.")
        print("\nTo run the applications later, follow these steps:")
        print(f"1. Activate the virtual environment:")
        print(f"   source {venv_path}/bin/activate")
        if run_flask:
            print(f"2. Start the Flask API:")
            print(f"   python {flask_script}")
        if run_streamlit:
            print(f"3. Start the Streamlit app:")
            print(f"   streamlit run {streamlit_script}")
