import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install


def _post_install():
    print('Running post-installation: downloading essential models...')
    try:
        subprocess.check_call([sys.executable, 'download_models.py'])
    except Exception as e:
        print(f'Warning: Model download failed: {e}')


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        _post_install()


class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        _post_install()


setup(
    name='idp_cv',
    version='0.1.0',
    description='Intelligent Document Processing and Computer Vision for Invoices',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Operating System :: POSIX :: Linux',
    ],
    python_requires='>=3.10, <3.14',
    install_requires=[
        'docling>=2.74.0',
        'spacy>=3.8.0',
        'sentence-transformers>=3.4.0',
        'transformers>=4.40.0',
        'onnxruntime>=1.15.0',
        'torch>=2.10.0',
        'torchvision>=0.25.0',
        'pandas>=2.2.0',
        'python-dateutil>=2.8.2',
        'python-dotenv>=1.0.0',
        'openpyxl>=3.1.2',
        'pillow>=10.2.0',
    ],
    extras_require={
        'dev': [
            'ipykernel>=6.29.0',
            'ruff',
            'pytest',
            'flake8',
            'black',
        ]
    },
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
    },
)
