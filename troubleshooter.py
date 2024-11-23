#!/usr/bin/env python3
import os
import sys
import logging
import subprocess
from pathlib import Path
import psutil
import platform
import re
import shutil
import time

# ----------------------------
# Setup Logging
# ----------------------------
def setup_logging():
    """
    Configure logging to output to both file and console.
    """
    logging.basicConfig(
        filename='troubleshooting.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)

# Initialize logging
setup_logging()

# ----------------------------
# Utility Functions
# ----------------------------
def find_virtualenv(directory):
    """
    Locate the virtual environment directory in the given directory.
    """
    for venv_name in ['.venv', 'venv']:
        venv_path = Path(directory) / venv_name
        if venv_path.exists():
            return venv_path
    return None

def check_virtualenv_dependencies(venv_path):
    """
    Check for issues in the virtual environment dependencies.
    """
    logging.info(f"Checking virtual environment at {venv_path}...")
    pip_executable = venv_path / "bin" / "pip" if platform.system() != "Windows" else venv_path / "Scripts" / "pip.exe"

    if not pip_executable.exists():
        logging.error("Pip executable not found in virtual environment.")
        print("Error: Pip executable not found in virtual environment.")
        return False

    try:
        result = subprocess.run(
            [str(pip_executable), "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            logging.error(f"Failed to list packages in virtual environment: {result.stderr.strip()}")
            print("Error: Unable to list packages in virtual environment.")
            return False
        logging.info("Virtual environment dependencies:")
        logging.info(result.stdout)
    except Exception as e:
        logging.error(f"Error checking virtual environment dependencies: {e}")
        return False
    return True

def reinstall_problematic_package(package_name, venv_path):
    """
    Attempt to reinstall a problematic package in the virtual environment.
    """
    pip_executable = venv_path / "bin" / "pip" if platform.system() != "Windows" else venv_path / "Scripts" / "pip.exe"
    try:
        subprocess.run(
            [str(pip_executable), "uninstall", "-y", package_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        subprocess.run(
            [str(pip_executable), "install", package_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logging.info(f"Successfully reinstalled {package_name}.")
    except Exception as e:
        logging.error(f"Failed to reinstall {package_name}: {e}")

def analyze_log_file(log_path):
    """
    Analyze the deployment log for potential issues.
    """
    if not os.path.exists(log_path):
        logging.error(f"Log file {log_path} does not exist.")
        return []

    issues_detected = {
        "missing_dependencies": False,
        "model_loading_errors": False,
        "slow_inference": False
    }

    with open(log_path, 'r') as file:
        for line in file:
            if "No module named" in line:
                missing_module = re.search(r"No module named '(.*?)'", line)
                if missing_module:
                    issues_detected["missing_dependencies"] = missing_module.group(1)
            if "Model loading failed" in line or "Exception occurred during model loading" in line:
                issues_detected["model_loading_errors"] = True
            if "Slow inference" in line:
                issues_detected["slow_inference"] = True

    return issues_detected

def suggest_fixes(issues_detected):
    """
    Suggest fixes for the identified issues.
    """
    fixes = []
    if issues_detected.get("missing_dependencies"):
        fixes.append(f"Install the missing dependency: {issues_detected['missing_dependencies']}")
    if issues_detected.get("model_loading_errors"):
        fixes.append("Review the model loading process. Ensure the correct configuration file is used.")
    if issues_detected.get("slow_inference"):
        fixes.append("Consider upgrading hardware or optimizing the model configuration.")
    return fixes

def check_system_resources(required_ram_gb=16):
    """
    Check if the system meets memory requirements.
    """
    total_ram = psutil.virtual_memory().total / (1024 ** 3)
    logging.info(f"Total RAM: {total_ram:.2f} GB")
    if total_ram < required_ram_gb:
        logging.warning(f"Insufficient RAM: {total_ram:.2f} GB available, but {required_ram_gb} GB required.")
        return False
    return True

def modify_deploy_script_for_fixes(deploy_script_path, issues_detected):
    """
    Modify the deploy script to address identified issues.
    """
    if not os.path.exists(deploy_script_path):
        logging.error(f"Deploy script {deploy_script_path} not found.")
        return False

    backup_path = deploy_script_path + '.bak'
    shutil.copy(deploy_script_path, backup_path)
    logging.info(f"Backup created: {backup_path}")

    with open(deploy_script_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        if issues_detected.get("missing_dependencies") and "import" in line:
            # Example fix: Add a comment near the missing dependency
            missing_dep = issues_detected["missing_dependencies"]
            if f"import {missing_dep}" in line:
                modified_lines.append(f"# Ensure {missing_dep} is installed\n")
        modified_lines.append(line)

    with open(deploy_script_path, 'w') as file:
        file.writelines(modified_lines)

    logging.info("Deploy script modified successfully.")
    return True

def redeploy_application(deploy_script_path):
    """
    Attempt to redeploy the application.
    """
    if not os.path.exists(deploy_script_path):
        logging.error(f"Deployment script {deploy_script_path} not found.")
        return False

    try:
        subprocess.check_call(['python3', deploy_script_path])
        logging.info("Application redeployed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to redeploy application: {e}")
        return False

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    log_path = "deployment_setup.log"
    deploy_script_path = "deploy.py"
    current_directory = Path.cwd()

    # Locate virtual environment
    venv_path = find_virtualenv(current_directory)
    if not venv_path:
        logging.error("Virtual environment not found. Ensure .venv or venv exists in the project directory.")
        sys.exit(1)

    # Check virtual environment dependencies
    check_virtualenv_dependencies(venv_path)

    # Analyze deployment log
    issues_detected = analyze_log_file(log_path)
    if not any(issues_detected.values()):
        logging.info("No issues detected in the log file.")
    else:
        logging.info("Issues detected:")
        for issue, details in issues_detected.items():
            logging.info(f" - {issue}: {details}")
        fixes = suggest_fixes(issues_detected)
        logging.info("Suggested fixes:")
        for fix in fixes:
            logging.info(f" - {fix}")

    # Check system resources
    if not check_system_resources():
        logging.warning("System does not meet the minimum memory requirements.")

    # Modify deploy script if needed
    if any(issues_detected.values()):
        modify_deploy_script_for_fixes(deploy_script_path, issues_detected)

    # Prompt user for redeployment
    response = input("Do you want to redeploy the application now? (y/n): ").strip().lower()
    if response in ["y", "yes"]:
        redeploy_application(deploy_script_path)