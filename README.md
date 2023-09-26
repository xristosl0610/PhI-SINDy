# PhI-SINDy
---
# Physics Informed Sparse Identification of Nonlinear Dynamics


## Table of contents
* [General info](#general-info)
* [Dependencies](#dependencies)
* [References](#references)
* [Contact](#contact)

## General info
Framework that builds on RK4-SINDy [1], including physics knowledge in the form of three different biases. Its goal is the identification of nonsmooth nonlinear systems where discontinuous functions contribute. The core application for the time being is the identification of friction forces in a dynamical (SDOF or MDOF) system. 

## Publication
A thorough step-by-step presentation of PhI-SINDy can be found in [Physics Enhanced Sparse Identification of Dynamical Systems with Discontinuous Nonlinearities](https://www.researchsquare.com/article/rs-3116800/v1)

## Technologies
Use the package manager [pip](https://pip.pypa.io/en/stable/) and the `requirements.txt` to install the necessary libraries

```bash
pip install -r requirements.txt
```
	
## References
[1]. P. Goyal, and P. Benner, [Discovery of Nonlinear Dynamical Systems using a Runge-Kutta Inspired Dictionary-based Sparse Regression Approach](https://arxiv.org/abs/2105.04869), arXiv:2105.04869, 2021.

[2]. S. L. Brunton, P. L. Proctor, and N. J. Kutz, [Discovering governing equations from data by sparse identification of nonlinear dynamical systems](https://doi.org/10.1073/pnas.1517384113), 2016.
<details><summary>BibTeX</summary><pre>
@article{goyal2022discovery,
  title={Discovery of nonlinear dynamical systems using a Runge--Kutta inspired dictionary-based sparse regression approach},
  author={Goyal, Pawan and Benner, Peter},
  journal={Proceedings of the Royal Society A},
  volume={478},
  number={2262},
  pages={20210883},
  year={2022},
  publisher={The Royal Society}
}
@article{brunton2016discovering,
  title={Discovering governing equations from data by sparse identification of nonlinear dynamical systems},
  author={Brunton, Steven L and Proctor, Joshua L and Kutz, J Nathan},
  journal={Proceedings of the national academy of sciences},
  volume={113},
  number={15},
  pages={3932--3937},
  year={2016},
  publisher={National Acad Sciences}
}
</pre></details>

## Contact
For further information, please contact xristosl0610@gmail.com
