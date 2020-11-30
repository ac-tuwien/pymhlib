import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymhlib",
    version="0.1.3",
    author="Günther Raidl et al.",
    author_email="raidl@ac.tuwien.ac.at",
    description="pymhlib - a toolbox for metaheuristics and hybrid optimization methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ac-tuwien/pymhlib",
    license='GPL3',
    packages=setuptools.find_packages(),
    package_data={"pymhlib":['demos/data/*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
