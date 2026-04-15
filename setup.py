from setuptools import setup, find_packages

setup(
    name="nlm",
    version="0.1.0",
    description="Neural Long Memory — hybrid long-term memory for AI agents",
    author="Hantes (Vitalii Halak)",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "sentence-transformers>=2.2.0",
        "chromadb>=0.4.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "gpu": ["torch>=2.0.0", "transformers>=4.30.0"],
        "dev": ["pytest>=7.0.0"],
    },
)
