import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="joml",
    version="0.8.7",
    author="Julien Jerphanion",
    author_email="julien.jerphanion@protonmail.com",
    description="A minimalist numpy-baked Neural Network API 🦎",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jjerphan/joml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
