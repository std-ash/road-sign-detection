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
        "torch",
        "torchvision",
        "pillow",
        "numpy",
        "opencv-python",
        "requests",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "tqdm"
    ],
)
