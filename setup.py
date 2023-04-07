import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="principal-feature-analysis",
    version="1.0.8",
    author="Tim Breitenbach & Lauritz Rasbach",
    author_email="tim.breitenbach@mathematik.uni-wuerzburg.de, rasbachlauritz@googlemail.com",
    description="The first package for Principal Feature Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=['numpy','scipy','pandas','networkx']
)
