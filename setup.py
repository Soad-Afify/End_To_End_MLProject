## This file will be responsable for creating the ML project as a package and also deploy it

from setuptools import find_packages, setup
from typing import List # we need this to build the below function

e_dot = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function is created to read all requiremed libraries in Requiremnets.txt file
    and convert it into a list to be handled in install_requires in setup function
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines() 
        requirements = [req.replace("\n", "") for req in requirements]

        if e_dot in requirements:
            requirements.remove(e_dot)
    return requirements


setup(

    name="ML_Project",
    version= "0.0.1",
    author="Soad_Afify",
    author_email="soad.afify@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements('Requirements.txt')

)
