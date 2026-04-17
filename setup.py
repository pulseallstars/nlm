from setuptools import setup, find_packages

setup(
    name="neural-long-memory",
    version="1.1.0",
    description="Neural Long Memory — hybrid long-term memory for AI agents",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Vitalii Halak",
    author_email="galakapp@gmail.com",
    url="https://github.com/pulseallstars/nlm",
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
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
