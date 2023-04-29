# from setuptools import setup, find_packages

# setup(
#     name="HapticMetadataGeneration",
#     version="0.1.0",
#     author="Brandon Nguyen",
#     author_email="brandonnguyen257@gmail.com",
#     description="Code for generating metadata for haptic data",
#     packages=find_packages(),
#     install_requires=[
#         "numpy",
#         "pandas",
#         "scikit-learn",
#     ],
# )


from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="HapticMetadataGeneration",
    version="0.1.0",
    author="Brandon Nguyen",
    author_email="brandonnguyen257@gmail.com",
    description="Code for generating metadata for haptic data",
    packages=find_packages(),
    install_requires=requirements,
)
