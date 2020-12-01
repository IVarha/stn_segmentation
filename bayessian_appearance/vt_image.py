

import vtk
import numpy as np

class Image:

    _image_instance = None
    _to_phys_mat = None
    _interpolation = None
    def __init__(self,filename):
        imr = vtk.vtkNIFTIImageReader()
        imr.SetFileName(filename)
        imr.Update()
        self._image_instance = imr.GetOutput()
        tr_mat = self._image_instance.GetIndexToPhysicalMatrix()

        self._to_phys_mat = np.zeros((4,4))

        self._to_phys_mat[0, 0] = tr_mat.GetElement(0, 0)
        self._to_phys_mat[0, 1] = tr_mat.GetElement(0, 1)
        self._to_phys_mat[0, 2] = tr_mat.GetElement(0, 2)
        self._to_phys_mat[0, 3] = tr_mat.GetElement(0, 3)

        self._to_phys_mat[1, 0] = tr_mat.GetElement(1, 0)
        self._to_phys_mat[1, 1] = tr_mat.GetElement(1, 1)
        self._to_phys_mat[1, 2] = tr_mat.GetElement(1, 2)
        self._to_phys_mat[1, 3] = tr_mat.GetElement(1, 3)

        self._to_phys_mat[2, 0] = tr_mat.GetElement(2, 0)
        self._to_phys_mat[2, 1] = tr_mat.GetElement(2, 1)
        self._to_phys_mat[2, 2] = tr_mat.GetElement(2, 2)
        self._to_phys_mat[2, 3] = tr_mat.GetElement(2, 3)

        self._to_phys_mat[3, 0] = tr_mat.GetElement(3, 0)
        self._to_phys_mat[3, 1] = tr_mat.GetElement(3, 1)
        self._to_phys_mat[3, 2] = tr_mat.GetElement(3, 2)
        self._to_phys_mat[3, 3] = tr_mat.GetElement(3, 3)

    def setup_bspline(self, num_spl):
        self._interpolation = vtk.vtkImageBSplineCoefficients()
        self._interpolation.SetSplineDegree(num_spl)
        self._interpolation.SetInputData(self._image_instance)
        self._interpolation.Update()


    def interpolate(self, vect):
        vect1 = np.array(list(vect)+ [1])

        phvect1 = np.dot(self._to_phys_mat, vect1)

        out = self._interpolation.Evaluate(list(phvect1[:3]))
        return out


