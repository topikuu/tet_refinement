# Library for local refinement of tetrahedral meshes
Reasonably fast algorithm for partial or "local" refinement of tetrahedral meshes for Python 3.7+. This algorithm emphasizes on performance over tetrahedron quality.

A closely related algorithm has been previously described by D.C. Thompson and P. P. Pébay in their article Embarrassingly Parallel Mesh Refinement by Edge Subdivision [[1]](#1)

The detailed explanation of this algorithm is included in `documentation/documentation.pdf`.

## Usage
```python
import numpy as np
from tet_refinement import refine

# Unit cube made of 5 tetrahedra
p = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
              [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
              [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
t = np.array([[0, 1, 1, 1, 2],
              [1, 2, 2, 4, 4],
              [2, 3, 4, 5, 6],
              [4, 7, 7, 7, 7]])

# Example 1:
# Split the center tetra and one of the side tetras.
tets_to_split = [2, 3]
(p2, t2, t2_to_t) = refine(p, t, marked_t=tets_to_split)
# p2 contains the new nodal positions, t2 contains the new tetrahedra, and
# t2_to_t maps columns of t2 to those of t, i.e. for each new tetrahedron, tells
# what was its parent tetrahedron. This is useful for e.g. mapping material
# parameters.


# Example 2:
# Split all the edges of the first tetra.
edges_to_split = np.array([[0, 0, 0, 1, 1, 2],
                           [1, 2, 4, 2, 4, 4]])
(p3, t3, t3_to_t) = refine(p, t, marked_e=edges_to_split)
```

## Dependencies
This library depends on [numpy](https://numpy.org/) and [scipy](https://www.scipy.org/).

## Licensing
This library is licensed under 3-clause BSD license. See `LICENSE`.

## Citations
<a id="1">[1]</a>
D. C. Thompson and P. P. Pébay, 
Embarrassingly parallel mesh refinement by edge subdivision
Engineering with Computers 2006, vol. 22, issue 2
DOI 10.1007/s00366-006-0020-3
