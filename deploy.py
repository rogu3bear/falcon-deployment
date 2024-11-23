#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
import logging
import importlib.util
import psutil
import time
import threading
import platform
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import signal
import socket

# Define required packages
required_packages = [
    "torch",
    "transformers",
    "flask",
    "streamlit",
    "psutil",
    "waitress",
    "rich",
    "requests"
]

# ----------------------------
# Setup Logging
# ----------------------------
def setup_logging():
    """
    Configure logging to output to both file and console.
    """
    logging.basicConfig(
        filename='deployment_setup.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')  # Simplify console output
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)

# Initialize logging
setup_logging()

# ----------------------------
# Utility Functions
# ----------------------------
def validate_python_installation():
    """
    Ensure Python 3 is installed and accessible.
    """
    try:
        subprocess.check_call(["python3", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        logging.error("Python 3 not found. Please install Python 3 and rerun this script.")
        print("Error: Python 3 not found. Please install Python 3 and rerun this script.")
        sys.exit(1)

def install_dependencies():
    """
    Install required dependencies.
    """
    try:
        logging.info("Installing dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade'] + required_packages)
        logging.info("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install dependencies: {e}")
        print(f"Error: Failed to install dependencies: {e}")
        sys.exit(1)

def validate_dependencies():
    """
    Check if all necessary dependencies are installed.
    """
    missing_packages = []
    for pkg in required_packages:
        if importlib.util.find_spec(pkg) is None:
            missing_packages.append(pkg)
    if missing_packages:
        logging.warning(f"Missing dependencies: {', '.join(missing_packages)}. Installing them now.")
        install_dependencies()
    else:
        # Additionally, ensure bitsandbytes is up-to-date if on Linux
        if platform.system() == 'Linux':
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'bitsandbytes'])
                logging.info("bitsandbytes is up-to-date.")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to update bitsandbytes: {e}")
                print(f"Error: Failed to update bitsandbytes: {e}")
                sys.exit(1)

def check_system_resources(required_ram_gb=16):
    """
    Check if the system has enough RAM.
    """
    total_ram = psutil.virtual_memory().total / (1024 ** 3)  # Convert bytes to GB
    available_ram = psutil.virtual_memory().available / (1024 ** 3)
    logging.info(f"Total RAM: {total_ram:.2f} GB")
    logging.info(f"Available RAM: {available_ram:.2f} GB")
    if total_ram < required_ram_gb:
        logging.error(f"Insufficient RAM: {total_ram:.2f} GB available, but {required_ram_gb} GB required.")
        print(f"Error: Insufficient RAM. You have {total_ram:.2f} GB, but {required_ram_gb} GB is required.")
        sys.exit(1)

def is_port_in_use(port):
    """
    Check if a port is in use by attempting to bind to it.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except OSError:
            return True

def prompt_change_port(port_name, default_port):
    """
    Prompt the user to enter a new port if the default is in use.
    """
    while True:
        try:
            new_port = int(input(f"Enter a new port for {port_name} (current {default_port}): ").strip())
            if not is_port_in_use(new_port):
                logging.info(f"Using port {new_port} for {port_name}.")
                return new_port
            else:
                print(f"Port {new_port} is also in use. Please try another port.")
        except ValueError:
            print("Please enter a valid port number.")

def setup_interactive():
    """
    Guide the user through interactive setup for model deployment.
    """
    logging.info("Starting Falcon 7B Deployment Setup Tool.")
    print("\n" + "="*50 + "\nFalcon 7B Deployment Setup\n" + "="*50)
    
    validate_python_installation()
    validate_dependencies()
    check_system_resources()
    
    # Define ports
    flask_port = 8500       # Flask API port
    streamlit_port = 8501   # Streamlit app port

    # Check if Flask port is in use
    if is_port_in_use(flask_port):
        print(f"Port {flask_port} is already in use.")
        flask_port = prompt_change_port("Flask API", flask_port)

    # Check if Streamlit port is in use
    if is_port_in_use(streamlit_port):
        print(f"Port {streamlit_port} is already in use.")
        streamlit_port = prompt_change_port("Streamlit app", streamlit_port)
    
    deploy_path = str(Path.cwd())
    run_flask = True
    run_streamlit = True
    
    # Interactive prompt for quantization
    while True:
        quantize_input = input("Do you want to apply dynamic quantization to reduce memory usage? (y/n): ").strip().lower()
        if quantize_input in ['y', 'yes']:
            quantize = True
            break
        elif quantize_input in ['n', 'no']:
            quantize = False
            break
        else:
            print("Please enter 'y' or 'n'.")

    # Interactive prompt for reserved memory adjustment
    while True:
        adjust_memory_input = input("Do you want to adjust the reserved memory (default is 2 GB)? (y/n): ").strip().lower()
        if adjust_memory_input in ['y', 'yes']:
            try:
                reserved_memory_gb = float(input("Enter the amount of RAM to reserve for OS and other processes (in GB, e.g., 2): ").strip())
                total_ram = psutil.virtual_memory().total / (1024 ** 3)  # Convert bytes to GB
                if reserved_memory_gb < 0 or reserved_memory_gb > total_ram:
                    print(f"Please enter a value between 0 and {total_ram:.2f} GB.")
                    continue
                reserved_memory = int(reserved_memory_gb * (1024 ** 3))
                break
            except ValueError:
                print("Please enter a valid number.")
        elif adjust_memory_input in ['n', 'no']:
            reserved_memory = 2 * 1024 ** 3  # 2 GB in bytes
            break
        else:
            print("Please enter 'y' or 'n'.")

    return deploy_path, "tiiuae/falcon-7b-instruct", run_flask, run_streamlit, flask_port, streamlit_port, quantize, reserved_memory

def setup_flask_api(model_name, flask_port, quantize, reserved_memory):
    """
    Generate and return the Flask API script as a string.
    """
    if quantize:
        quantization_code = """        # Apply dynamic quantization
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        logging.info("Dynamic quantization applied.")
"""
    else:
        quantization_code = """        logging.info("Proceeding without quantization.")
"""

    flask_api_content = f"""#!/usr/bin/env python3
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading
import logging
import time
import sys
import platform
import traceback
import psutil
from transformers import BitsAndBytesConfig

app = Flask(__name__)

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')  # Simplify console output
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)

model = None
tokenizer = None
model_ready = False

def load_model():
    global model, tokenizer, model_ready
    try:
        start_time = time.time()
        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('{model_name}')

        logging.info("Loading model...")
        os_name = platform.system()
        use_bitsandbytes = False

        if os_name == 'Linux':
            try:
                import bitsandbytes as bnb
                use_bitsandbytes = True
                logging.info("bitsandbytes is available. Attempting to load model with 4-bit quantization.")
            except ImportError:
                logging.warning("bitsandbytes is not installed or not working. Proceeding without quantization.")
        else:
            logging.warning(f"Operating System '{{os_name}}' is not supported for bitsandbytes. Proceeding without quantization.")

        # Adjust max_memory based on system resources
        total_ram = psutil.virtual_memory().total
        reserved_memory = {reserved_memory}  # Reserved memory in bytes
        max_memory = total_ram - reserved_memory

        max_memory_dict = {{}}
        if torch.cuda.is_available():
            max_memory_dict['cuda:0'] = int(max_memory)
            device_map = 'auto'
        else:
            max_memory_dict['cpu'] = int(max_memory)
            device_map = 'cpu'

        if use_bitsandbytes:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                '{model_name}',
                quantization_config=bnb_config,
                device_map=device_map,
                max_memory=max_memory_dict
            )
        else:
            # Load model without bitsandbytes
            model = AutoModelForCausalLM.from_pretrained(
                '{model_name}',
                device_map=device_map,
                max_memory=max_memory_dict,
                low_cpu_mem_usage=True
            )
{quantization_code}
        model.eval()
        model_ready = True
        end_time = time.time()
        loading_time = end_time - start_time
        logging.info(f"Model loaded in {{loading_time / 60:.2f}} minutes.")
    except Exception as e:
        logging.error(f"Model loading failed: {{e}}")
        logging.exception("Exception occurred during model loading:")
        sys.exit(1)

@app.route('/status', methods=['GET'])
def status():
    if model_ready:
        return jsonify({{'status': 'ready'}})
    else:
        return jsonify({{'status': 'loading'}}), 202

@app.route('/generate', methods=['POST'])
def generate():
    if not model_ready:
        return jsonify({{'error': 'Model is not ready yet.'}}), 503
    data = request.get_json()
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({{'error': 'No prompt provided.'}}), 400
    try:
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({{'response': response}})
    except Exception as e:
        logging.error(f"Error during generation: {{e}}")
        logging.exception("Exception occurred during text generation:")
        return jsonify({{'error': 'Error during generation.'}}), 500

if __name__ == '__main__':
    threading.Thread(target=load_model, daemon=True).start()
    from waitress import serve
    serve(app, host='0.0.0.0', port={flask_port})
"""
    return flask_api_content

def setup_streamlit_app(flask_port):
    """
    Generate and return the Streamlit app script as a string.
    """
    streamlit_app_content = f"""#!/usr/bin/env python3
import streamlit as st
import requests
import time

st.title("Falcon 7B Interactive Interface")

status_placeholder = st.empty()
prompt_input = st.empty()
response_output = st.empty()

api_url = f"http://localhost:{flask_port}"

def check_status():
    try:
        response = requests.get(f"{{api_url}}/status")
        status = response.json().get('status', 'unknown')
        return status
    except Exception as e:
        return f"Error: {{e}}"

def wait_for_model():
    with st.spinner("Waiting for the model to load..."):
        while True:
            status = check_status()
            if status == 'ready':
                status_placeholder.success("Model is ready.")
                break
            elif isinstance(status, str) and status.startswith("Error"):
                status_placeholder.error(f"API status: {{status}}. Retrying in 5 seconds...")
            else:
                status_placeholder.info(f"Model status: {{status}}. Checking again in 5 seconds...")
            time.sleep(5)

wait_for_model()

def generate_response():
    prompt = prompt_input.text_area("Enter your prompt:")
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                try:
                    response = requests.post(f"{{api_url}}/generate", json={{"prompt": prompt}})
                    if response.status_code == 200:
                        result = response.json()
                        response_output.text_area("Response:", value=result.get("response", ""), height=200)
                    else:
                        error_msg = response.json().get("error", "Unknown error")
                        st.error(f"Error: {{error_msg}}")
                except Exception as e:
                    st.error(f"Failed to connect to API: {{e}}")
        else:
            st.warning("Please enter a prompt.")

generate_response()
"""
    return streamlit_app_content

def write_script(script_content, script_name, deploy_path):
    """
    Write the given script content to a file.
    """
    script_path = os.path.join(deploy_path, script_name)
    with open(script_path, "w") as script_file:
        script_file.write(script_content)
    os.chmod(script_path, 0o755)
    logging.info(f"{script_name} script created at {script_path}")
    return script_path

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    deploy_path, model_name, run_flask, run_streamlit, flask_port, streamlit_port, quantize, reserved_memory = setup_interactive()

    # Initialize variables
    flask_process = None
    streamlit_process = None
    system_load_thread = None
    stop_event = threading.Event()

    try:
        if run_flask:
            # Generate Flask API script content
            flask_api_content = setup_flask_api(model_name, flask_port, quantize, reserved_memory)
            # Write Flask API script to app.py
            flask_script_path = write_script(flask_api_content, "app.py", deploy_path)
            # Start the Flask API as a subprocess
            logging.info("Starting Flask API...")
            flask_process = subprocess.Popen(['python3', flask_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logging.info(f"Flask API started on port {flask_port}.")

            # Start a thread to capture Flask API logs
            def capture_flask_logs(proc):
                try:
                    for line in iter(proc.stdout.readline, b''):
                        if line:
                            logging.debug(f"Flask: {line.decode().strip()}")
                except Exception as e:
                    logging.error(f"Error capturing Flask stdout: {e}")
                try:
                    for line in iter(proc.stderr.readline, b''):
                        if line:
                            logging.error(f"Flask Error: {line.decode().strip()}")
                except Exception as e:
                    logging.error(f"Error capturing Flask stderr: {e}")

            flask_log_thread = threading.Thread(target=capture_flask_logs, args=(flask_process,), daemon=True)
            flask_log_thread.start()

            # Wait for the model to be ready
            logging.info("Waiting for the model to be ready...")
            while True:
                try:
                    response = requests.get(f"http://localhost:{flask_port}/status", timeout=5)
                    if response.status_code == 200 and response.json().get('status') == 'ready':
                        logging.info("Model is ready.")
                        break
                except requests.exceptions.RequestException:
                    pass
                time.sleep(5)

        if run_streamlit:
            # Generate Streamlit app script content
            streamlit_app_content = setup_streamlit_app(flask_port)
            # Write Streamlit app script to streamlit_app.py
            streamlit_script_path = write_script(streamlit_app_content, "streamlit_app.py", deploy_path)
            # Start the Streamlit app as a subprocess
            logging.info("Starting Streamlit app...")
            streamlit_process = subprocess.Popen(['streamlit', 'run', streamlit_script_path, '--server.port', str(streamlit_port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logging.info(f"Streamlit app started on port {streamlit_port}.")

            # Start a thread to capture Streamlit logs
            def capture_streamlit_logs(proc):
                try:
                    for line in iter(proc.stdout.readline, b''):
                        if line:
                            logging.debug(f"Streamlit: {line.decode().strip()}")
                except Exception as e:
                    logging.error(f"Error capturing Streamlit stdout: {e}")
                try:
                    for line in iter(proc.stderr.readline, b''):
                        if line:
                            logging.error(f"Streamlit Error: {line.decode().strip()}")
                except Exception as e:
                    logging.error(f"Error capturing Streamlit stderr: {e}")

            streamlit_log_thread = threading.Thread(target=capture_streamlit_logs, args=(streamlit_process,), daemon=True)
            streamlit_log_thread.start()

        print("\nPress Ctrl+C to stop the applications.")
        logging.info("Press Ctrl+C to stop the applications.")

        # Use 'rich' for advanced console output
        from rich.console import Console
        from rich.table import Table

        console = Console()

        def display_system_load():
            """
            Display system CPU and memory usage in a formatted table.
            """
            while not stop_event.is_set():
                cpu_load = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                table = Table(show_header=False, header_style="bold magenta")
                table.add_row("CPU Usage:", f"{cpu_load}%")
                table.add_row("Memory Usage:", f"{memory.percent}%")
                table.add_row("Available Memory:", f"{memory.available / (1024 * 1024):.2f} MB")
                console.clear()
                console.print(table)
                time.sleep(4)

        # Start the system load display in a separate thread
        system_load_thread = threading.Thread(target=display_system_load, daemon=True)
        system_load_thread.start()

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logging.info("Shutting down...")
        print("\nShutting down...")
        stop_event.set()  # Signal threads to stop

        # Terminate Flask API subprocess if running
        if flask_process and flask_process.poll() is None:
            flask_process.terminate()
            try:
                flask_process.wait(timeout=10)
                logging.info("Flask API terminated.")
            except subprocess.TimeoutExpired:
                flask_process.kill()
                logging.warning("Flask API subprocess killed after timeout.")

        # Terminate Streamlit app subprocess if running
        if streamlit_process and streamlit_process.poll() is None:
            streamlit_process.terminate()
            try:
                streamlit_process.wait(timeout=10)
                logging.info("Streamlit app terminated.")
            except subprocess.TimeoutExpired:
                streamlit_process.kill()
                logging.warning("Streamlit app subprocess killed after timeout.")

        # Additional cleanup for threading
        if system_load_thread and system_load_thread.is_alive():
            system_load_thread.join(timeout=5)

        sys.exit(0)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"Error: An unexpected error occurred: {e}")
        sys.exit(1)
