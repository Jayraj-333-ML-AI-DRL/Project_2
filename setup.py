
from setuptools import find_packages, setup
# Import necessary modules from setuptools

HYPEN_DOT_E = "-e ."
# Constant representing the string "-e .", used for checking requirements

def get_requirements(filepath: str):
    """Returns a list of requirements from the given file path."""
    # Function to read requirements from a file and return them as a list
    requirements = []
    with open(file=filepath, mode='r') as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        # Read each line from the file, removing newline characters
    if HYPEN_DOT_E in requirements:
        # Check if '-e .' is in the requirements list
        requirements.remove(HYPEN_DOT_E)
        # If present, remove it from the list
    return requirements
    # Return the list of requirements

setup(
    name='Project 2',
    version='1.0',
    # Description is commented out, as it's not provided here
    author='Jayraj Rajput',
    author_email='jayrajput1997@gmail.com',
    packages=find_packages(),
    # Find all packages automatically
    install_requirements=get_requirements('requirements.txt')
    # Specify requirements using the function defined above
)
