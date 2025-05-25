from setuptools import setup, find_packages

setup(
    name="road-sign-detection",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask",
        "flask-cors",
        "gunicorn",
        # CPU-only PyTorch to reduce size
        "torch==1.9.0+cpu",
        "torchvision==0.10.0+cpu",
        "pillow",
        "numpy",
        "opencv-python-headless",  # Headless version to reduce size
        "requests",
        # Minimal dependencies for YOLOv5
        "pandas",
        "tqdm"
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html"
    ],
)
