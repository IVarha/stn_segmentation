import vtk
import numpy as np
import os

class Mesh:
    _mesh_instance = None
    _points = None
    _triangles = None

    _connectivity_list = None
    _copied_points = None
    _triangles_copy = None

    _pt_size = None

    _file_name = None

    def calculate_copy_of_triangles(self):
        res = []
        for cell_id in range(self._mesh_instance.GetNumberOfCells()):
            a = vtk.vtkIdList()
            self._triangles.GetCellAtId(cell_id, a)
            id_list = []
            for id in range(a.GetNumberOfIds()):
                id_list.append(a.GetId(id))
            res.append(id_list)
        self._triangles_copy = np.array(res)

    def calculate_copy_of_points(self):
        self._copied_points = np.array(self.get_all_points())

    def calculate_closest_points(self):
        mat = []

        for pt_id in range(self._points.GetNumberOfPoints()):
            t_connected = []
            for cell_id in range(self._mesh_instance.GetNumberOfCells()):
                a = vtk.vtkIdList()
                self._triangles.GetCellAtId(cell_id, a)
                id_list = []
                for id in range(a.GetNumberOfIds()):
                    id_list.append(a.GetId(id))

                if pt_id in id_list:
                    for id in id_list:
                        if id != pt_id:
                            t_connected.append(id)
            mat.append(list(set(t_connected)))
        self._connectivity_list = mat

    def __init__(self, filename):

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
        # calculate options
        self.calculate_closest_points()
        self.calculate_copy_of_points()
        self.calculate_copy_of_triangles()
        self._pt_size = self._points.GetNumberOfPoints()
        self._file_name = filename


    def apply_transform(self, mat):
        for i in range(self._pt_size):
            point = list(self._points.GetPoint(i))
            pt_res = np.dot(mat, point + [1])
            self._points.SetPoint(i, tuple(pt_res[:3]))
        self._mesh_instance.Initialize()
        self._mesh_instance.SetPoints(self._points)
        self._mesh_instance.SetPolys(self._triangles)

    def get_min_max(self):

        pts = []
        for i in range(self._pt_size):
            pts.append(list(self._points.GetPoint(i)))
        pts = np.array(pts)

        res = [pts[:, 0].min(), pts[:, 0].max(),  # xmin xmax
               pts[:, 1].min(), pts[:, 1].max(),  # ymin ymax
               pts[:, 2].min(), pts[:, 2].max(), ]  # zmin zmax
        return res

    def points_is_inside(self, pts_3d):

        points = vtk.vtkPoints()
        for i in range(pts_3d.shape[0]):
            for j in range(pts_3d.shape[1]):
                for k in range(pts_3d.shape[2]):
                    points.InsertNextPoint(tuple(pts_3d[i, j, k, :]))
        pd = vtk.vtkPolyData()
        pd.SetPoints(points)

        encl_points = vtk.vtkSelectEnclosedPoints()
        encl_points.SetInputData(pd)
        encl_points.SetSurfaceData(self._mesh_instance)
        encl_points.Update()

        result = np.zeros([pts_3d.shape[0], pts_3d.shape[1], pts_3d.shape[2]])
        cnt = 0
        for i in range(pts_3d.shape[0]):
            for j in range(pts_3d.shape[1]):
                for k in range(pts_3d.shape[2]):
                    ind = cnt
                    val = encl_points.IsInside(ind)
                    result[i, j, k] = val
                    cnt += 1
        return result

        pass

    def generate_normals(self, mm_len, npts):
        dt = (2 * mm_len) / (npts - 1)
        nm_del = [-mm_len + dt * x for x in range(npts)]

        norm_calc = vtk.vtkPolyDataNormals()
        norm_calc.ComputeCellNormalsOff()
        norm_calc.ComputePointNormalsOn()
        norm_calc.SetAutoOrientNormals(True)
        norm_calc.SetInputData(self._mesh_instance)
        norm_calc.Update()

        normals = ((norm_calc.GetOutput()).GetPointData()).GetNormals()
        res = []
        for i in range(self._points.GetNumberOfPoints()):

            tmp = []
            norm = np.array(normals.GetTuple(i))
            norm = norm / np.linalg.norm(norm)

            pt = np.array(self._points.GetPoint(i))

            for j in range(len(nm_del)):
                poin = pt + nm_del[j] * norm
                tmp.append(list(poin))
            res.append(tmp)
        return res

    def get_point(self, i):
        return self._points.GetPoint(i)

    def gen_num_of_points(self):
        return self._points.GetNumberOfPoints()

    def get_all_points(self):

        res = []
        for i in range(self.gen_num_of_points()):
            res.append(list(self._points.GetPoint(i)))
        return res

    def get_unpacked_coords(self):
        res = []
        for i in range(self.gen_num_of_points()):
            res += (list(self._points.GetPoint(i)))
        return res

    def modify_points(self, coords):
        coords = np.array(coords)
        for i in range(coords.shape[0]):
            self._points.SetPoint(i,tuple(coords[i,:]))
        self._mesh_instance.Initialize()
        self._mesh_instance.SetPoints(self._points)
        self._mesh_instance.SetPolys(self._triangles)

    def save_obj(self, filename):
        try:
            os.remove(filename)
        except:
            pass
        writer = vtk.vtkOBJWriter()
        writer.SetFileName(filename)
        writer.SetInputData(self._mesh_instance)
        writer.Write()

    def is_triangle_intercepted(self, V10, V11, V12, V20, V21, V22):
        N2 = np.cross(V21 - V20, V22 - V20)
        d2 = -np.dot(N2, V20)
        eq =0
        if ((V10 == V20).min() == True) or ((V10 == V21).min() == True) or ((V10 == V22).min() == True): eq+=1
        if ((V11 == V20).min() == True) or ((V11 == V21).min() == True) or ((V11 == V22).min() == True): eq += 1
        if ((V12 == V20).min() == True) or ((V12 == V21).min() == True) or ((V12 == V22).min() == True): eq += 1

        if eq == 1:
            return False

        dist10 = np.dot(N2, V10) + d2

        dist11 = np.dot(N2, V11) + d2

        dist12 = np.dot(N2, V12) + d2

        if (((dist10 >= 0) and (dist11 >= 0) and (dist12 >= 0)) or ((dist10 <= 0) and (dist11 <= 0) and (dist12 <= 0))):
            return False
        if ((dist10 == 0) or (dist11 == 0) or (dist12 == 0)):
            return False
        else:
            N1 = np.cross((V11 - V10), (V12 - V10))

            d1 = -np.dot(N1, V10)

            dist20 = np.dot(N1, V20) + d1
            dist21 = np.dot(N1, V21) + d1
            dist22 = np.dot(N1, V22) + d1
            if (((dist20 >= 0) and (dist21 >= 0) and (dist22 >= 0)) or (
                    (dist20 <= 0) and (dist21 <= 0) and (dist22 <= 0))):
                return False
            D = np.cross(N1, N2)

            p10 = np.dot(D, V10)

            p11 = np.dot(D, V11)

            p12 = np.dot(D, V12)


            p20 = np.dot(D, V20)

            p21 = np.dot(D, V21)

            p22 = np.dot(D, V22)
            if (abs(dist10) < 1e-4): dist10=0
            if (abs(dist11) < 1e-4): dist11=0
            if (abs(dist12) < 1e-4): dist12=0
            if (abs(dist20) < 1e-4): dist20=0
            if (abs(dist21) < 1e-4): dist21=0
            if (abs(dist22) < 1e-4): dist22=0

            if ((dist20 == 0) or  (dist21 == 0) or  (dist22 == 0)):
                return False
            if ((dist10 - dist12 == 0)):
                return False
            if (((dist10 >= 0) and (dist11 >= 0)) or ((dist10 <= 0) and (dist11 <= 0))):

                t11 = p10 - (p10 - p12) * (dist10 / (dist10 - dist12))
                t12 = p11 - (p11 - p12) * (dist11 / (dist11 - dist12))
            elif (((dist10 >= 0) and (dist12 >= 0)) or ((dist10 <= 0) and (dist12 <= 0))):
                t11 = p10 - (p10 - p11) * (dist10 / (dist10 - dist11))
                t12 = p12 - (p12 - p11) * (dist12 / (dist12 - dist11))
            else:
                t11 = p11 - (p11 - p10) * (dist11 / (dist11 - dist10))
                t12 = p12 - (p12 - p10) * (dist12 / (dist12 - dist10))

            if (((dist20 >= 0) and (dist21 >= 0)) or ((dist20 <= 0) and (dist21 <= 0))):

                t21 = p20 - (p20 - p22) * (dist20 / (dist20 - dist22))
                t22 = p21 - (p21 - p22) * (dist21 / (dist21 - dist22))
            elif (((dist20 >= 0) and (dist22 >= 0)) or ((dist20 <= 0) and (dist22 <= 0))):

                t21 = p20 - (p20 - p21) * (dist20 / (dist20 - dist21))
                t22 = p22 - (p22 - p21) * (dist22 / (dist22 - dist21))
            else:
                t21 = p21 - (p21 - p20) * (dist21 / (dist21 - dist20))
                t22 = p22 - (p22 - p20) * (dist22 / (dist22 - dist20))

            if   ( ((t21<=t11) and (t21<=t12) and  (t22<=t11) and (t22<=t12)) or
                    ((t21 >= t11) and (t21 >= t12) and (t22 >= t11) and (t22 >= t12))):
                return False
        return True


    def calculate_interception_from_newPTS(self, newPts):
        newPts = newPts.reshape((self._points.GetNumberOfPoints(), 3))
        j_s = 1
        for i in range(self._triangles_copy.shape[0]):
            v1_i = self._triangles_copy[i, 0]
            v2_i = self._triangles_copy[i, 1]
            v3_i = self._triangles_copy[i, 2]

            v11 = newPts[v1_i, :]
            v21 = newPts[v2_i, :]
            v31 = newPts[v3_i, :]

            for j in range(j_s, self._pt_size):
                v12_i = self._triangles_copy[j, 0]
                v22_i = self._triangles_copy[j, 1]
                v32_i = self._triangles_copy[j, 2]

                v12 = newPts[v12_i, :]
                v22 = newPts[v22_i, :]
                v32 = newPts[v32_i, :]

                if (self.is_triangle_intercepted(v11, v21, v31, v12, v22, v32)):
                    return -1
        return 1
