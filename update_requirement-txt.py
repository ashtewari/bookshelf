import pkg_resources
import subprocess

# Define requirements file path
requirements_file = "requirements_new.txt"

# Load existing package names from requirements.txt while preserving order
try:
    with open(requirements_file, "r") as f:
        # Keep original order by using a list instead of a set
        required_packages = [line.strip().split("==")[0] for line in f if line.strip()]
except FileNotFoundError:
    print(f"Error: {requirements_file} not found.")
    exit(1)

# Get currently installed packages
installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

# Match installed versions with required packages, maintaining original order
updated_requirements = [
    f"{pkg}=={installed_packages[pkg]}"
    for pkg in required_packages if pkg in installed_packages
]

# Write updated requirements back to file
if updated_requirements:
    with open(requirements_file, "w") as f:
        f.write("\n".join(updated_requirements) + "\n")
    print(f"✅ {requirements_file} has been updated with installed package versions.")
else:
    print("⚠️ No matching installed packages found for the existing requirements.")

