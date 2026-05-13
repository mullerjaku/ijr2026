from setuptools import setup, find_packages

setup(
    name="iwinac2026",
    version="0.0.1",
    author="Jakub Müller",
    description="Intrinsic motivation methods for goal exploration",
    # find_packages() automaticky najde složku 'classes' a další, pokud mají __init__.py
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        "numpy",
        "torch",
        "gymnasium",
        "gymnasium-robotics",
        "mujoco",  
        "matplotlib",
        "scikit-image",     
        "opencv-python",    
        "ollama",
        "scikit-learn",   
        "joblib",
    ],
)