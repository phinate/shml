# shml

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![Code style: black][black-badge]][black-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]
[![Gitter][gitter-badge]][gitter-link]

# `shml`: routines to automate machine learning experiments for a X -> SH -> bbyy search

This module aims to provide a set of functions that, when composed, can run a pipeline capable of:
- going from `.root` files to `parquet` files via `uproot` and `awkward`
- constructing useful kinematic quantites for training
- applying a chosen or manual preselection
- configuring any additional processing, e.g. weight normalization, feature scaling
- access event data that's prepared for `pytorch` using `shml.torch_dataset.EventDataset`

still to do:

- infra to run ml experiments in a GPU or CPU environment via `pytorch-lightining`

## Install
For preprocessing only:
```
python3 -m pip install shml
```
For ML extras:
```
python3 -m pip install shml[ml]
```
[actions-badge]:            https://github.com/phinate/shml/workflows/CI/badge.svg
[actions-link]:             https://github.com/phinate/shml/actions
[black-badge]:              https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]:               https://github.com/psf/black
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/shml
[conda-link]:               https://github.com/conda-forge/shml-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/phinate/shml/discussions
[gitter-badge]:             https://badges.gitter.im/https://github.com/phinate/shml/community.svg
[gitter-link]:              https://gitter.im/https://github.com/phinate/shml/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
[pypi-link]:                https://pypi.org/project/shml/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/shml
[pypi-version]:             https://badge.fury.io/py/shml.svg
[rtd-badge]:                https://readthedocs.org/projects/shml/badge/?version=latest
[rtd-link]:                 https://shml.readthedocs.io/en/latest/?badge=latest
[sk-badge]:                 https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
