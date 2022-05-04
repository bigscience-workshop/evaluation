from setuptools import find_packages, setup


def req_file(filename):
    with open(filename) as f:
        content = f.readlines()
    return [x.strip() for x in content]


install_requires = req_file("requirements.txt")

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="evaluation",
    python_requires=">=3.8.11,<4.0",  # Some dependencies need explicit "<4.0"
    version="0.1.1",
    url="https://github.com/bigscience-workshop/evaluation.git",
    author="Multiple Authors",
    author_email="xxx",
    description="",
    long_description=readme,
    packages=find_packages(),
    install_requires=install_requires,
)
