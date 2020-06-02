import setuptools

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name="duelpy",
    version="0.0.1",
    author="The duelpy team",
    author_email="contact.us.at@gitlab.invalid",
    description="Dueling Bandit Algorithms in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/duelpy/duelpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
