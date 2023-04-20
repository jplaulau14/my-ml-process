from setuptools import setup, find_packages

setup(
    name="my-ml-process",
    version="0.1.0",
    description="My Machine Learning Process",
    author="John Patrick Laurel",
    author_email="prod.patricklaurel@gmail.com",
    url="https://github.com/jplaulau14/my-ml-process",
    packages=find_packages(include=["project"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "imblearn"
    ],
    python_requires='>=3.6',
)