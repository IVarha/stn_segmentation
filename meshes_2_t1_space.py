import pickle
import sys

import ExtPy
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mesh_file = sys.argv[1]
    transformation_file = sys.argv[2]

    outp = sys.argv[3]

    f = open(transformation_file,'rb')
    transf = pickle.load(f)

    mesh = ExtPy.cMesh(mesh_file)
    mesh.apply_transform(np.linalg.inv(transf))
    mesh.save_obj(outp)


