from setuptools import setup, find_packages

HYPEN_E_DOT = '-e .'

def get_requirements(file_path)->list:
    requirements = []
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name='ML PROJECT',
    version='0.0.1',
    author='Vaibhav C Poojari',
    author_email='vaibhavpoojari5@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirement.txt')
    )