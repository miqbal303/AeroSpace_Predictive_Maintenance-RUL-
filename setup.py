from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT= '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_object:
        requirements = file_object.readlines()
        requirements = [req.replace("\n","") for req in requirements]


        if HYPEN_E_DOT in requirements:
            requirements = requirements.remove(HYPEN_E_DOT)
        
    return requirements

setup(
    name="Turbo_Engine_Predict_Maintenance",
    version = '0.0.1',
    author ='MIqbal',
    author_email = 'miqbal@gmail.com',
    install_requires = get_requirements('requirements.txt'),
    packages = find_packages()
    )