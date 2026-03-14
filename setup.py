# setup.py para instalação local
from setuptools import setup, find_packages

setup(
    name="little-hawk",
    version="0.1.0",
    description="LLM streaming engine em NumPy puro",
    author="tiagouzl",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "huggingface_hub",
        "tokenizers",
        "fastapi",
        "pydantic"
    ],
    python_requires=">=3.8",
)
