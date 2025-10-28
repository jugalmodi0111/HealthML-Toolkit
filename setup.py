"""
HealthML-Toolkit setup configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="healthml-toolkit",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="End-to-end ML toolkit for healthcare: conversational AI, data quality, and medical imaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/HealthML-Toolkit",
    packages=find_packages(exclude=["tests", "docs", "configs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "nltk>=3.8.0",
        "tensorflow>=2.15.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "kaggle>=1.5.0",
        "jupyter>=1.0.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "pdf": [
            "reportlab>=4.0.0",
        ],
        "apple-silicon": [
            "tensorflow-macos>=2.15.0",
            "tensorflow-metal>=1.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "healthml=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="healthcare machine-learning medical-imaging chatbot data-imputation explainable-ai grad-cam diabetic-retinopathy",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/HealthML-Toolkit/issues",
        "Source": "https://github.com/yourusername/HealthML-Toolkit",
        "Documentation": "https://github.com/yourusername/HealthML-Toolkit/tree/main/docs",
    },
)
