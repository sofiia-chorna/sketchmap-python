from setuptools import setup, find_packages

setup(
    name="sketchmap_python",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "click",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "torch",
        "tqdm",
    ],
    author="Sofiia Chorna",
    entry_points={"console_scripts": ["sketchmap_python-cli=sketchmap_python.cli:cli"]},
)
