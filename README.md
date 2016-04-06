# MagDots: Magnetization dynamics of nanoelement arrays

[![DOI](https://zenodo.org/badge/19702/ivlis/MagDots.svg)](https://zenodo.org/badge/latestdoi/19702/ivlis/MagDots)

## Purpose

The program is capable to calculate a spin-wave spectrum and a distribution of an internal magnetic
field in infinite and semi-infinite arrays of magnetic elements arranged in period lattice with a
complex elementary cell. The program implements ideas proposed in ["Theoretical formalism for
collective spin-wave edge excitations in arrays of dipolarly interacting magnetic
nanodots"](http://arxiv.org/abs/1511.08483).

## Installation

### Requirements
* Python 3.3 or later
* Cython and a C compiler (gcc)
* Make
* NumPy and SciPy
* GNU GSL library from your distribution (headers and binary)
* Matplotlib for plotting

To compile the Cython code execute the Makefile:
`make`

Compilation was tested on Linux-based and OS X systems only,
however, should work on other systems with Cython and a C compiler installed. 

To run tests type `make tests`

Having the Cython code complied, one can run examples from the `examples` directory

## Usage

The program has two modes for bulk and edge spectra calculations. Examples in the `examples`
directory are self-explanatory, but it is implied that a user is familiar with the method of calculation.

### Examples

#### Bulk spin-wave modes
* `bulk/calc_FM.py` calculation of *bulk* spin-wave spectra of an array of circular pillars arranged in a
  square lattice in a ferromagnetic ground state
* `bulk/calc_CAFM.py` the same as for the above, but for a chessboard antiferromagnetic ground state
* `bulk/calc_equilibrium.py` calculate spin-wave spectra of an array of nanoelements arranged in a
  hexagonal lattice , magnetized by a bias magnetic field in an arbitrary direction

#### Edge spin-wave modes
* `edge/calc_FM.py` calculation of *edge* spin-wave spectra of an array of circular pillars arranged in a
  square lattice in a ferromagnetic ground state
* `edge/calc_CAFM.py` the same as for the above, but for a chessboard antiferromagnetic ground state
* `edge/calc_CAFM_domain_wall.py` calculation of a domain-wall spin-wave spectrum existing between
  two chessboard antiferromagnetic ground states

### How to cite 

If you are using this program or/and produce scientific publications based on it,
we kindly ask you cite it as:

> Ivan Lisenkov, Vasyl Tyberkevych, Sergey Nikitov, Andrei Slavin  
> Theoretical formalism for collective spin-wave edge excitations in arrays of dipolarly interacting magnetic nanodots  
> arXiv:1511.08483  
> http://arxiv.org/abs/1511.08483


### Copyright

The program is written by Ivan Lisenkov. The analytical theory is developed by Ivan Lisenkov, Vasyl
Tyberkevych and Andrei Slavin at The Department of Physics, Oakland University, Rochester, Michigan, USA.

(C) 2015 Ivan Lisenkov

Department of Physics  
Oakland University  
2200 N. Squirrel Rd  
Rochester, MI, U.S.A.  
48309-4401

MagDots is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MagDots is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MagDots.  If not, see <http://www.gnu.org/licenses/>.



