from setuptools import find_packages, setup
from typing import List
def get_requirements(file_path:str)->List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements
setup(
    name = "ml_project",
    version = "0.1.0",
    author = "elifrsakn",
    author_email = "elifsakins@gmail.com",
    install_requires = ["pandas", "numpy", "scikit-learn", "matplotlib", "seaborn"]

    
)