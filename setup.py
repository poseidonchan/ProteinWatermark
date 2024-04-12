from setuptools import setup, find_packages
setup(
    name = 'protein-watermark',
    version = '0.1',
    description = 'adding watermark to generative protein design models',
    author = 'Yanshuo Chen',
    author_email = 'alegendaryfish@icloud.com',
    url = 'https://github.com/poseidonchan/ProteinWatermark',
    license = 'Apache-2.0 License',
    packages = find_packages(),
    python_requires='>=3.8',
    platforms = 'any',
    install_requires = [
        'numpy',
        'scipy',
        'transformers',
    ],
)