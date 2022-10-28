# Introducing topography in convolutional neural networks

![tests](https://github.com/mxmpl/topography/actions/workflows/tests.yml/badge.svg?branch=main)
![linting](https://github.com/mxmpl/topography/actions/workflows/linting.yml/badge.svg?branch=main)
![python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
![os](https://img.shields.io/badge/OS-Linux%20%7C%20MacOS%20%7C%20Windows-green)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## Installation

```bash
conda create -n topography pip=22 python=3.10
conda activate topography
pip install .
```

```bash
conda create -n topography-dev pip=22 python=3.10
conda activate topography-dev
pip install -e .[dev,testing]
```

`pip` version at least 21.3 for the editable install.
