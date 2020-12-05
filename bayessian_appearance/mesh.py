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
        cnt = 0
        for i in range(pts_3d.shape[0]):
            for j in range(pts_3d.shape[1]):
                for k in range(pts_3d.shape[2]):
                    ind = cnt
                    val = encl_points.IsInside(ind)
                    result[i,j,k] = val
                    cnt+=1
        return result


        pass


    def generate_normals(self,mm_len, npts):
        dt = (2* mm_len) / (npts - 1)
        nm_del = [ -mm_len + dt*x for x in range(npts)]

        norm_calc = vtk.vtkPolyDataNormals()
        norm_calc.ComputeCellNormalsOff()
        norm_calc.ComputePointNormalsOn()
        norm_calc.SetAutoOrientNormals(True)
        norm_calc.SetInputData(self._mesh_instance)
        norm_calc.Update()

        normals = ((norm_calc.GetOutput()).GetPointData()).GetNormals()
        res  = []
        for i in range(self._points.GetNumberOfPoints()):

            tmp = []
            norm = np.array(normals.GetTuple(i))
            norm = norm /np.linalg.norm(norm)

            pt = np.array(self._points.GetPoint(i))

            for j in range(len(nm_del)):
                poin = pt + nm_del[j] * norm
                tmp.append(list(poin))
            res.append(tmp)
        return res

    def get_point(self, i):
        return self._points.GetPoint(i)