from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="graphius",
    version="0.0.1",
    description="A simple graph library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alperen ÜNLÜ",
    author_email="alperenunlu.cs@gmail.com",
    url="https://github.com/alperenunlu/graphius",
    packages=find_packages(),
    license="MIT",
    install_requires=["graphviz", "pillow"],
)
