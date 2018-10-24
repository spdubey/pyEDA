from setuptools import setup, find_packages

__version__ = '1.0.0'

setup(
    name='pyEDA',
    version=__version__,
    packages=find_packages(),
    url='https://generalmills.visualstudio.com/ACE/_git/pyEDA',
    license='Apache License 2.0',
    author='X39192A',
    author_email='abhishek.dubey@genmills.com',
    description='Creating basic exploratory data analysis report ',
    include_package_data=True,
    install_requires=['Seaborn','missingno'],
    keywords=['eda'],
    classifiers=[
        "Development Status :: 1 - Planning",
        # "Development Status :: 2 - Pre-Alpha",
        # "Development Status :: 3 - Alpha",
        # "Development Status :: 4 - Beta",
        # "Development Status :: 5 - Production/Stable",
        # "Development Status :: 6 - Mature",
        # "Development Status :: 7 - Inactive",        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: Implementation :: PyPi",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ]
)
