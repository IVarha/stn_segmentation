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
    _points_in_triangles = None

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

        res_tri = []
        for i in range(self.gen_num_of_points()):
            p_nb = []
            for j in range(self._triangles_copy.shape[0]):
                if i in list(self._triangles_copy[j,:]):
                    p_nb.append(j)
            res_tri.append(p_nb)
        self._points_in_triangles = res_tri



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

    def __init__(self, filename=None):
        if filename != None:
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
        norm_calc.SetInputData(self._mesh_instance)
        norm_calc.ComputeCellNormalsOff()
        norm_calc.ComputePointNormalsOn()
        norm_calc.SetAutoOrientNormals(True)

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


    def get_centre(self):
        cn  = vtk.vtkCenterOfMass()
        cn.SetInputData(self._mesh_instance)
        cn.Update()
        return cn.GetCenter()

    @staticmethod
    def generate_sphere(center, radius,discretisation):
        ss = vtk.vtkSphereSource()
        ss.SetRadius(radius)
        ss.SetCenter(center[0],center[1],center[2])
        ss.Update()
        originalMesh = ss.GetOutput()
        sdf = vtk.vtkLinearSubdivisionFilter()
        sdf.SetInputData(originalMesh)
        sdf.SetNumberOfSubdivisions(discretisation)
        sdf.Update()
        mes = sdf.GetOutput()
        res = Mesh()
        res._points = mes.GetPoints()
        res._triangles = mes.GetPolys()
        res._mesh_instance = mes
        res.calculate_closest_points()
        res.calculate_copy_of_points()
        res.calculate_copy_of_triangles()
        return res





    def _interpolate_value_vox(self,mask,vox):

        if vox == None: return 0
        x = vox[0]
        y = vox[1]
        z = vox[2]


        x1 = np.floor(vox[0])

        y1 = np.floor(vox[1])

        z1 = np.floor(vox[2])

        xd = vox[0] - x1

        yd = vox[1] - y1

        zd = vox[2] - z1

        if (((vox[0] + 1) > mask.shape[0]) or ((y + 1) > mask.shape[1]) or ((z + 1) > mask.shape[2]) or (x < 0) or (
                y < 0) or (z < 0)): return 0;

        c000 = mask[(int)(x1), (int)(y1), (int)(z1)]
        c001 = mask[int(x1), int(y1), int(z1 + 1)]

        c010 = mask[int(x1), int(y1 + 1), int(z1)]

        c011 = mask[int(x1), int(y1 + 1), int(z1 + 1)]

        c100 = mask[int(x1 + 1), int(y1), int(z1)]

        c101 = mask[int(x1 + 1), int(y1), int(z1 + 1)]

        c110 = mask[int(x1 + 1), int(y1 + 1), int(z1)]

        c111 = mask[int(x1 + 1), int(y1 + 1), int(z1 + 1)]


        c00 = c000 * (1 - xd) + c100 * xd

        c01 = c001 * (1 - xd) + c101 * xd

        c10 = c010 * (1 - xd) + c110 * xd

        c11 = c011 * (1 - xd) + c111 * xd


        c0 = c00 * (1 - yd) + c10 * yd

        c1 = c01 * (1 - yd) + c11 * yd

        return c0 * (1 - zd) + c1 * zd


    #for sphere shrinkage
    def _recalc_to_voxel_coords(self, coords,mni_coords):
        tx=0
        if ((mni_coords[0] > coords[0][-1])
                or(mni_coords[1] > coords[1][-1])
                or (mni_coords[2] > coords[2][-1])
                or (mni_coords[0] < coords[0][0])
                or (mni_coords[1] < coords[1][0])
                or (mni_coords[2] < coords[2][0])
        ):
            return None
        #for x
        tx = 0
        for i in range(len(coords[0])-1):
            if (coords[0][i]<mni_coords[0]) and (coords[0][i+1]>mni_coords[0]):
                tx = i
                break
        tx =tx+ (mni_coords[0] - coords[0][tx])/(coords[0][tx+1] - coords[0][tx])

        #for y
        ty = 0
        for i in range(len(coords[1])-1):
            if (coords[1][i]<mni_coords[1]) and (coords[1][i+1]>mni_coords[1]):
                ty = i
                break
        ty =ty+ (mni_coords[1] - coords[1][ty]) / (coords[1][ty + 1] - coords[1][ty])
        # for z
        tz = 0
        for i in range(len(coords[2]) - 1):
            if (coords[2][i] < mni_coords[2]) and (coords[2][i + 1] > mni_coords[2]):
                tz = i
                break
        tz = tz + (mni_coords[2] - coords[2][tz]) / (coords[2][tz + 1] - coords[2][tz])
        return [tx,ty,tz]


    def _move_point(self, image,point, coords, direction, stop, reach_value, step, eps):
        curr_pt=point

        cur_dir = 1

        curr_step = step

        val_pre = self._interpolate_value_vox(image, self._recalc_to_voxel_coords(coords,curr_pt))


        err = abs(val_pre - reach_value)
        while (err > eps):

            val1 = self._interpolate_value_vox(image, self._recalc_to_voxel_coords(coords,curr_pt + curr_step*cur_dir*direction))

            t_err = np.linalg.norm( curr_pt - stop)

            if (t_err < curr_step):
                break

            curr_pt = curr_pt + curr_step*cur_dir*direction

            if (((val1 > reach_value) and (val_pre < reach_value)) or ((val1 < reach_value) and (val_pre > reach_value))):

                curr_step = curr_step / 2
                cur_dir = cur_dir * (-1)

            else:
                if (val1 == reach_value):
                    break
            err = abs(val1 - reach_value)
            val_pre = val1

        return curr_pt

    def shrink_sphere(self, mask, coords,cenre,tres):
        cenre = np.array(cenre)
        for i in range(self._copied_points.shape[0]):

            normal = cenre - self._copied_points[i,:]
            normal = normal/np.linalg.norm(normal)


            #move+point
            res_vox = self._move_point(mask,point=self._copied_points[i,:],coords=coords,
                                       direction=normal,stop=cenre,reach_value=tres,step=0.3,eps=0.03)

            self._points.SetPoint(i,res_vox)
            print("processed point " + str(i))

        self._mesh_instance.Initialize()
        self._mesh_instance.SetPoints(self._points)
        self._mesh_instance.SetPolys(self._triangles)


    def smooth_mesh(self,iterations):
        sf = vtk.vtkSmoothPolyDataFilter()
        sf.SetInputData(self._mesh_instance)
        sf.SetNumberOfIterations(iterations)
        sf.FeatureEdgeSmoothingOff()
        sf.BoundarySmoothingOn()
        sf.SetRelaxationFactor(0.1)
        sf.Update()

        self._points.Initialize()
        self._points.DeepCopy(sf.GetOutput().GetPoints())
        self._mesh_instance.Initialize()
        self._mesh_instance.SetPoints(self._points)
        self._mesh_instance.SetPolys(self._triangles)


    def triangle_normalisation(self, iterations, fraction):

        for i in range(iterations):

            for pt in range(self._copied_points.shape[0]):
                areas = []
                for tri in range(len(self._points_in_triangles[pt])):
                    triangle = self._points_in_triangles[pt][tri]
                    pt1 = self._triangles_copy[triangle,0]
                    pt2 = self._triangles_copy[triangle,1]
                    pt3 = self._triangles_copy[triangle,2]

                    pt_1 = np.array(self._points.GetPoint(pt1))
                    pt_2 = np.array(self._points.GetPoint(pt2))
                    pt_3 = np.array(self._points.GetPoint(pt3))

                    areas.append(abs(np.dot( (pt_2 - pt_1).transpose(),pt_3-pt_1 )))

                areas = np.array(areas)
                max_i = np.where( areas == max(areas))[0][0]
                max_tri = self._points_in_triangles[pt][max_i]

                ind_p = pt
                ind_o1 = None
                ind_o2 = None
                if self._triangles_copy[max_tri,0] == pt:
                    ind_o1 = self._triangles_copy[max_tri,1]
                    ind_o2 = self._triangles_copy[max_tri, 2]
                elif self._triangles_copy[max_tri,1] == pt:
                    ind_o1 = self._triangles_copy[max_tri,0]
                    ind_o2 = self._triangles_copy[max_tri, 2]
                else:
                    ind_o1 = self._triangles_copy[max_tri,0]
                    ind_o2 = self._triangles_copy[max_tri, 1]

                point = np.array(self._points.GetPoint(pt))

                a1 = np.array(self._points.GetPoint(ind_o1))
                a2 = np.array(self._points.GetPoint(ind_o2))

                mid = (a1 + a2)/2
                dir = (point - mid)*fraction
                self._points.SetPoint(pt,tuple(point - dir))

        self.update_mesh()

    def update_mesh(self):
        self._mesh_instance.Initialize()
        self._mesh_instance.SetPoints(self._points)
        self._mesh_instance.SetPolys(self._triangles)


    def lab_move_points(self, mask, threshold,coords):
        normalsGen = vtk.vtkPolyDataNormals()
        normalsGen.SetInputData(self._mesh_instance)
        normalsGen.ComputeCellNormalsOff()
        normalsGen.ComputePointNormalsOn()
        normalsGen.SetAutoOrientNormals(True)
        normalsGen.Update()

        normals = normalsGen.GetOutput()
        normals4 = normals.GetPointData().GetNormals()


        for i in range(self.gen_num_of_points()):
            vox = self._points.GetPoint(i)

            normal = normals4.GetTuple(i)

            res_vox = self.move_in_value_dir(vox,mask,np.array(normal),0.1,threshold,coords)
            self._points.SetPoint(i, tuple(res_vox))
        self._mesh_instance.Initialize()
        self._mesh_instance.SetPoints(self._points)
        self._mesh_instance.SetPolys(self._triangles)

    def move_in_value_dir(self,point, mask, direction, step, threshold,coords):
        curr_pt = point


        val_pre = self._interpolate_value_vox(mask, self._recalc_to_voxel_coords(coords, curr_pt))

        val1 = self._interpolate_value_vox(mask, self._recalc_to_voxel_coords(coords,
                                                        curr_pt + step * direction))

        val2 = self._interpolate_value_vox(mask, self._recalc_to_voxel_coords(coords,
                                                        curr_pt - step * direction))

        if (abs(val1 - threshold) < abs(val_pre - threshold)):
            return curr_pt + step * direction

        elif abs(val1 - 1) < 0.1:
            return curr_pt + step * direction


        return curr_pt

    # generates point array
    @staticmethod
    def generate_mask(x_r, y_r, z_r, dt):

        x_min = x_r[0]
        x_max = x_r[1]

        x_arr = []
        i = 0
        while True:
            x_arr.append(x_min + i * dt)
            if (x_min + i * dt) >= x_max:
                break
            i += 1
        y_min = y_r[0]
        y_max = y_r[1]
        y_arr = []
        i = 0
        while True:
            y_arr.append(y_min + i * dt)
            if (y_min + i * dt) >= y_max:
                break
            i += 1

        z_min = z_r[0]
        z_max = z_r[1]
        z_arr = []
        i = 0
        while True:
            z_arr.append(z_min + i * dt)
            if (z_min + i * dt) >= z_max:
                break
            i += 1

        res_arr = np.zeros((len(x_arr), len(y_arr), len(z_arr), 3))
        for i in range(len(x_arr)):
            for j in range(len(y_arr)):
                for k in range(len(z_arr)):
                    res_arr[i, j, k, 0] = x_arr[i]
                    res_arr[i, j, k, 1] = y_arr[j]
                    res_arr[i, j, k, 2] = z_arr[k]
        return res_arr


    @staticmethod
    def meshes_overlap(mesh_main, mesh_test, method):


        pt_m  = mesh_main.get_all_points()

        pt_test = mesh_test.get_all_points()

        pt_c = np.array(pt_m + pt_test)

        min_X = pt_c[:,0].min()
        max_X = pt_c[:, 0].max()
        min_Y = pt_c[:,1].min()
        max_Y = pt_c[:, 1].max()
        min_Z = pt_c[:,2].min()
        max_Z = pt_c[:, 2].max()


        point_array = Mesh.generate_mask([min_X,max_X],[min_Y,max_Y],[min_Z,max_Z],0.2)

        mask_main = mesh_main.points_is_inside(point_array)
        mask_test = mesh_test.points_is_inside(point_array)


        return  method(mask_main,mask_test)


