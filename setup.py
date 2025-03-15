from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        # Remove '-e .' if present in requirements.txt

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="Hand gesture drawing proejct",  # Project name
    version="0.0.1",  # Version number
    author="Nasim",
    author_email="nasimakram6200@gmail.com",
    install_requires=get_requirements("requirements.txt"),  # Read dependencies from requirements.txt
    packages=find_packages()  # Automatically find and include all packages in the current directory
)
