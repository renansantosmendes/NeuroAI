import subprocess
import os

try:
    result = subprocess.run(
        ["python", "-m", "pytest", "--cov=src", "--cov-report=term", "tests/"],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
except Exception as e:
    print(f"Error: {e}")
