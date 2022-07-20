from setuptools import find_packages, setup

setup(
    name='mprnn',
    packages=find_packages(
        include = ['utils','training','testing','compare_nets',
                    'baseline_agents']
    ),
)