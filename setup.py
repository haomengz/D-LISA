from setuptools import find_packages, setup

setup(
    name="dlisa",
    py_modules=["dlisa"],
    version="1.0",
    author="Haomeng Zhang",
    description="D-LISA",
    packages=find_packages(include=("dlisa*")),
    install_requires=[
        "torch==2.0.1",
        "numpy==1.26.0",
        f"clip @ git+ssh://git@github.com/eamonn-zh/CLIP.git", 
        "lightning==2.0.6", 
        "wandb==0.15.8", 
        "scipy", 
        "hydra-core",
        "h5py", 
        "pandas"
    ]
)