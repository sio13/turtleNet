
import setuptools

from toggl_api.__version__ import version



with open('README.md', 'r') as fh:

    long_description = fh.read()



setuptools.setup(

    name='turtleNet',

    version=version,

    author='Martin Gano',

    author_email='ganomartin@gmail.com',

    description='Framework for experimenting with adversarial attacks and defences',

    long_description=long_description,

    long_description_content_type='text/markdown',

    url='https://github.com/sio13/turtleNet',

    packages=setuptools.find_packages(),

    classifiers=[

        "Programming Language :: Python :: 3",

        "License :: OSI Approved :: MIT License",

        "Operating System :: OS Independent",

    ],

    python_requires='>=3.6',

)