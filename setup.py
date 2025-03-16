from setuptools import setup, find_packages
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='medvqa',
    version='0.12',
    packages=find_packages() +
    find_packages(include=['*'], where='./competitions/**/'),
    entry_points={
        'console_scripts': [
            'medvqa=medvqa.cli:main',
        ],
    },
    install_requires=[
        # Add your dependencies here
    ],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
