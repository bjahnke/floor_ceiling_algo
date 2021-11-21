from setuptools import setup

setup(
    name='trade_utils',
    version='5.15.2021.0',
    packages=[''],
    url='',
    license='',
    author='Brian',
    author_email='',
    description='',
    install_requires=[
        'openpyxl',
        'websocket-client',
        'better-abc',
        'dotmap',
        'numpy',
        'matplotlib',
        'pandas',
        'selenium',
        'scipy',
        'httpx',
        'tda-api',
        'yfinance',
        'schedule',
        'w3rw @ git+https://github.com/teleprint-me/w3rw.git#egg=w3rw',
        'coinbase-pro @ git+https://github.com/teleprint-me/coinbase-pro.git#egg=coinbase-pro'
    ]
)
