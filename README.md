chainer-dense-fusion
====================

[![Build Status](https://travis-ci.com/knorth55/chainer-dense-fusion.svg?branch=master)](https://travis-ci.com/knorth55/chainer-dense-fusion)

<img src="_static/example.png" width="75%" />

This is [Chainer](https://github.com/chainer/chainer) implementation of [DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion](https://arxiv.org/abs/1901.04780).

Original PyTorch repository is [j96w/DenseFusion](https://github.com/j96w/DenseFusion).

Requirement
-----------

- [CuPy](https://github.com/cupy/cupy)
- [Chainer](https://github.com/chainer/chainer)
- [ChainerCV](https://github.com/chainer/chainercv)
- OpenCV

Installation
------------

We recommend to use [Anacoda](https://anaconda.org/).

```bash
# Requirement installation
conda create -n dense-fusion python=3.6
source activate dense-fusion 
pip install opencv-python
pip install cupy

# Installation
git clone https://github.com/knorth55/chainer-dense-fusion.git
cd chainer-dense-fusion/
pip install -e .
```

Inference
---------

```bash
cd examples/dense_fusion/
python demo.py --random
```

TODO
----
- YCB Video Dataset
  - [x] Add estimator inference script.
  - [x] Add refiner inference script.
  - [ ] Add training script.
  - [ ] Reproduce original accuracy.

LICENSE
-------
[MIT LICENSE](LICENSE)
