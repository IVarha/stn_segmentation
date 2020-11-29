import vtk
import numpy as np

class Mesh:

    _mesh_instance = None
    _points = None
    _triangles = None

    def __init__(self,filename):

        reader = vtk.vtkOBJReader()
        reader.SetFileName(filename)
        reader.Update()

        a = reader.GetOutput()

        self._mesh_instance = vtk.vtkPolyData()

        self._points = vtk.vtkPoints()
        self._triangles = vtk.vtkCellArray()

        self._points.DeepCopy(a.GetPoints())
        self._triangles.DeepCopy(a.GetPolys())

        self._mesh_instance.SetPoints(self._points)
        self._mesh_instance.SetPolys(self._triangles)


    def apply_transform(self, mat):
        for i in range(self._points.GetNumberOfPoints()):
            point = list(self._points.GetPoint(i))
            pt_res = np.dot(mat, point + [1])
            self._points.SetPoint(i, tuple(pt_res[:3]))
        self._mesh_instance.Initialize()
        self._mesh_instance.SetPoints(self._points)
        self._mesh_instance.SetPolys(self._triangles)


    def get_min_max(self):

        pts = []
        for i in range(self._points.GetNumberOfPoints()):
            pts.append(list(self._points.GetPoint(i)))
        pts = np.array(pts)

        res = [pts[:,0].min(),pts[:,0].max(), #xmin xmax
               pts[:,1].min(),pts[:,1].max(), #ymin ymax
               pts[:,2].min(),pts[:,2].max(),] #zmin zmax
        return res


    def points_is_inside(self, pts_3d):

        points = vtk.vtkPoints()
        for i in range(pts_3d.shape[0]):
            for j in range(pts_3d.shape[1]):
                for k in range(pts_3d.shape[2]):
                    points.InsertNextPoint(tuple(pts_3d[i,j,k,:]))
        pd = vtk.vtkPolyData()
        pd.SetPoints(points)

        encl_points = vtk.vtkSelectEnclosedPoints()
        encl_points.SetInputData(pd)
        encl_points.SetSurfaceData(self._mesh_instance)
        encl_points.Update()

        result = np.zeros([pts_3d.shape[0],pts_3d.shape[1],pts_3d.shape[2]])
        for i in range(pts_3d.shape[0]):
            for j in range(pts_3d.shape[1]):
                for k in range(pts_3d.shape[2]):
                    ind = i*pts_3d.shape[1]*pts_3d.shape[2] + k*pts_3d.shape[2] + k
                    val = encl_points.IsInside(ind)
                    result[i,j,k] = val
        return result


        pass



