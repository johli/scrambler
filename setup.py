import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scrambler",
    version="0.1",
    author="Johannes Linder",
    author_email="johannes.linder@hotmail.com",
    description="Deep Generative Masking for Sequences",
    long_description=long_description,
    url="https://github.com/johli/scrambler",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
