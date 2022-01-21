import setuptools
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()


setuptools.setup(
    name="hate_target",
    version="0.0.0",
    description="Package for predicting target identity of hate speech.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta"
    ]
)
