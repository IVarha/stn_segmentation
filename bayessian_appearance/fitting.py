

import bayessian_appearance.point_distribution as pd
import bayessian_appearance.mesh as mesh
import bayessian_appearance.utils as utils

import numpy as np
import bayessian_appearance.vt_image as vtimage
import fsl.data.image as fim
import fsl.transform.flirt as fl
import os
import nibabel as nib
import bayessian_appearance.settings as settings
from datetime import datetime

import sys

sys.path.insert(0, "/tmp/bayessian_segmentation_cpp/cmake-build-debug-remote-host")
import ExtPy


import bayessian_appearance.settings as gl_set

import scipy.optimize as opt
#for single mesh
class FunctionHandler:

    _image = None
    _from_mni_to_vox = None
    _vox_to_world = None
    #label of single mesh
    _label = None
    _mesh = None
    _cmesh = None
    _num_of_points=None


    _constraints = None
    _kdes = None

    def __init__(self):
        pass

    def set_image(self, filename):

        vt = vtimage.Image(filename)
        vt.setup_bspline(3)
        self._image = vt

        a = nib.load(filename)
        self._vox_to_world = a.affine


    #TODO rewrite to make it universal
    def set_subject(self,subject_dir):

        mni_im = fim.Image(subject_dir + os.sep + "t1_brain_to_mni_stage2_apply.nii.gz" )

        native_im = fim.Image(subject_dir + os.sep + "t1_acpc_extracted.nii.gz" )

        from_mni_mat = fl.readFlirt(subject_dir + os.sep + "combined_affine_reverse.mat")


        mni_w = fl.fromFlirt(from_mni_mat,mni_im,native_im,"world","world")
        a = np.linalg.inv(self._vox_to_world)
        self._from_mni_to_vox = np.dot(a,mni_w)


    def __call__(self, *args, **kwargs):
        coords = np.array(args[0])


        coords = self._kdes[0].vector_2_points(coords)
        coords2 = coords.reshape((self._num_of_points,3))

        #print(datetime.now())
        #self._mesh.modify_points(coords2)

        #normals = self._mesh.generate_normals(gl_set.settings.norm_length,gl_set.settings.discretisation)
        #print(datetime.now())
        self._cmesh.modify_points(coords)
        normals = self._cmesh.generate_normals(gl_set.settings.norm_length,gl_set.settings.discretisation)
        #print(datetime.now())
        mesh_pts = utils.apply_transf_2_pts(self._cmesh.generate_mesh_points(10),transf=self._from_mni_to_vox)

        normals = utils.apply_transf_2_norms(normals,self._from_mni_to_vox)


        ips = np.array(self._image.interpolate_list(mesh_pts)).mean()


        norm_intens = np.array(self._image.interpolate_normals(normals))

        #normalise intensity
        norm_intens  = norm_intens - ips

        norm_intens = norm_intens.reshape((norm_intens.shape[0]*norm_intens.shape[1]))
        #distr_coords = np.concatenate((coords,norm_intens))

        return - self._kdes[1](coords,norm_intens)



#for multi mesh
class FunctionHandlerMulti:

    _image = None
    _from_mni_to_vox = None
    _vox_to_world = None
    #label of single mesh
    _label = None
    _mesh = None
    _cmesh = None
    _num_of_points=None


    _constraints = None
    _kdes = None

    def __init__(self):
        pass

    def set_image(self, filename):

        vt = vtimage.Image(filename)
        vt.setup_bspline(3)
        self._image = vt

        a = nib.load(filename)
        self._vox_to_world = a.affine


    #TODO rewrite to make it universal
    def set_subject(self,subject_dir):

        mni_im = fim.Image(subject_dir + os.sep + "t1_brain_to_mni_stage2_apply.nii.gz" )

        native_im = fim.Image(subject_dir + os.sep + "t1_acpc_extracted.nii.gz" )

        from_mni_mat = fl.readFlirt(subject_dir + os.sep + "combined_affine_reverse.mat")


        mni_w = fl.fromFlirt(from_mni_mat,mni_im,native_im,"world","world")
        a = np.linalg.inv(self._vox_to_world)
        self._from_mni_to_vox = np.dot(a,mni_w)


    def __call__(self, *args, **kwargs):
        coords = np.array(args[0])


        coords = self._kdes.vector_2_points(coords)
        #coords2 = coords.reshape((self._num_of_points,3))

        #print(datetime.now())
        #self._mesh.modify_points(coords2)

        #normals = self._mesh.generate_normals(gl_set.settings.norm_length,gl_set.settings.discretisation)
        #print(datetime.now())
        nits = None
        for i in range(self._constraints):

            self._cmesh.modify_points(coords[i*3*self._num_of_points:(i+1)*3*self._num_of_points ])


            normals = self._cmesh.generate_normals(gl_set.settings.norm_length,gl_set.settings.discretisation)
        #print(datetime.now())
            mesh_pts = utils.apply_transf_2_pts(self._cmesh.generate_mesh_points(10),transf=self._from_mni_to_vox)

            normals = utils.apply_transf_2_norms(normals,self._from_mni_to_vox)


            ips = np.nanmean(np.array(self._image.interpolate_list(mesh_pts)))


            norm_intens = np.array(self._image.interpolate_normals(normals))

        #normalise intensity
            norm_intens  = norm_intens - ips

            norm_intens = norm_intens.reshape((norm_intens.shape[0]*norm_intens.shape[1]))
            if nits is None:
                nits = norm_intens
            else:
                nits = np.concatenate((nits,norm_intens))
        #distr_coords = np.concatenate((coords,norm_intens))

        return - self._kdes(coords,nits)

















class Fitter:

    _pdm = None

    _train_subj = None
    _test_subj = None
    _modal = None
    _mat_over = None
    _best_meshes_mni = None
    _best_meshes_c = None



    def __init__(self,train_subj,test_subj,pdm=None):
        self._train_subj = train_subj
        self._test_subj = test_subj
        if pdm != None:
            if isinstance(pdm,str):
                self._pdm = pd.PointDistribution.read_pdm(file_name=pdm)

            else:
                self._pdm = pdm

    def read_pdm(self, filename ):
        self._pdm = pd.PointDistribution.read_pdm(file_name=filename)


    def set_modalities(self, modalities):
        self._modal = modalities


    def set_overlaped(self,file_name):
        mat = np.loadtxt(file_name)
        self._mat_over = mat

        #get best meshes of segmented figures
        self._best_meshes_mni = []
        self._best_meshes_c = []
        for i in range(len(gl_set.settings.labels_to_segment)):


            label = gl_set.settings.labels_to_segment[i]
            # ind = gl_set.settings.all_labels.index(label)
            #
            # sub_ind = list(self._mat_over[ind]).index(1)
            # mni_im = fim.Image(self._train_subj[sub_ind] + os.sep + "t1_brain_to_mni_stage2_apply.nii.gz")
            #
            # native_im = fim.Image(self._train_subj[sub_ind] + os.sep + "t1_acpc_extracted.nii.gz")
            #
            # to_mni_mat = fl.readFlirt(self._train_subj[sub_ind] + os.sep + "combined_affine_t1.mat")
            #
            # mni_w = fl.fromFlirt(to_mni_mat, native_im, mni_im, "world", "world")
            #
            # surf_c = ExtPy.cMesh(self._train_subj[sub_ind] + os.sep + label + "_1.obj")
            #surf= mesh.Mesh(self._train_subj[sub_ind] + os.sep + label + "_1.obj")
            surf_c = ExtPy.cMesh(gl_set.settings.atlas_dir + os.sep + label + "m.obj")
            surf = mesh.Mesh(gl_set.settings.atlas_dir + os.sep + label + "m.obj")
            # surf.apply_transform(mni_w)
            # surf_c.applyTransformation(mni_w.tolist())
            self._best_meshes_c.append(surf_c)
            self._best_meshes_mni.append(surf)




    def fit_single(self):
        cds = self._pdm.recompute_conditional_shape_int_distribution(self._best_meshes_mni[0].gen_num_of_points())

        for i_test_sub in range(len(self._test_subj)):

            for lab in range(len(self._best_meshes_mni)):
                fc = FunctionHandler()
                #todo fix set
                fc.set_image(  self._test_subj[i_test_sub] + os.sep + "t2_acpc_normalised.nii.gz/t2_resampled_fcm.nii.gz"   )
                fc.set_subject(self._test_subj[i_test_sub])
                fc._mesh = self._best_meshes_mni[lab]
                fc._cmesh = self._best_meshes_c[lab]

                fc._mesh.calculate_closest_points()
                fc._kdes = cds[lab]
                fc._num_of_points= self._best_meshes_mni[lab].gen_num_of_points()

                #constraint for interception
                int_Cons = opt.NonlinearConstraint(fc._mesh.calculate_interception_from_newPTS,lb=0,ub=2)


                X0 = fc._mesh.get_unpacked_coords()
                X0 = cds[lab][0].decompose_coords_to_eigcords(X0)
                X0[:] = 0
                #####SAVE initi
                res_points = cds[lab][0].vector_2_points(X0)
                l = len(res_points.tolist())
                fc._mesh.modify_points(res_points.reshape(( int(l/3),3)))
                fc._mesh.apply_transform(utils.read_fsl_mni2native_w(self._test_subj[i_test_sub]))
                fc._mesh.save_obj(self._test_subj[i_test_sub] + os.sep + self._pdm._label_kde[lab] + "_initialise.obj")

                ##########################################################
                print(datetime.now())
                print(fc._cmesh.selfIntersectionTest(list(cds[lab][0].vector_2_points(X0))))
                print(datetime.now())
                #print(fc._mesh.calculate_interception_from_newPTS(np.array(X0)))
                print(datetime.now())


                bounds = cds[lab][0].generate_bounds(3)
                #mimiser = opt.minimize(fc,X0,method="cg",options={"disp":True})

                # Xpt = X0.copy()
                # for bd in range(1,len(bounds)+1):
                #     bound = bounds[-bd]
                #
                #     curr = bound[0]
                #     arr = []
                #     dt = bound[1]*2/100
                #     cnt = 0
                #     while (bound[0] + dt*cnt < bound[1]):
                #         Xpt[-bd] = bound[0] + dt*cnt
                #         arr.append(fc(Xpt))
                #         cnt = cnt+1
                #
                #     arr = np.array(arr)
                #     ind = np.where( arr == min(arr))[0][0]
                #     Xpt[-bd] = bound[0] +ind*dt
                #
                # X0 = Xpt


                print("Start optimising ")
                mimiser = opt.minimize(fc, X0, method='TNC',bounds=bounds, options={"disp": True})
                r_x = mimiser.fun
                while True:
                    mimiser = opt.minimize(fc, mimiser.x, method='TNC', bounds=bounds, options={"disp": True, 'ftol': 1})
                    mimiser = opt.minimize(fc, mimiser.x, method='Powell', bounds=bounds,
                                          options={"disp": True
                                              # ,'ftol':1
                                                   })

                    if r_x - mimiser.fun < 0.1:
                        break
                    r_x = mimiser.fun
                mimiser = opt.minimize(fc, mimiser.x, method='CG',tol=1, options={"disp": True,
                                                                                  # 'ftol': 1
                                                                                  })
                r_x = mimiser.fun
                # mimiser = opt.minimize(fc, X0, method='L-BFGS-B', bounds=bounds, options={"disp": True})
                # mimiser = opt.minimize(fc, X0, method="cg", options={"disp": True})

                #mimiser = opt.minimize(fc, X0,method='COBYLA',tol=1,constraints=con,options={"maxiter":5000})
                print(datetime.now())
                res_points = cds[lab][0].vector_2_points(mimiser.x)
                l = len(res_points.tolist())
                fc._mesh.modify_points(res_points.reshape(( int(l/3),3)))
                fc._mesh.apply_transform(utils.read_fsl_mni2native_w(self._test_subj[i_test_sub]))

                fc._mesh.save_obj(self._test_subj[i_test_sub] + os.sep + self._pdm._label_kde[lab] + "_fitted.obj")

                a = fc(X0)
            pass
        pass



    def fit_multiple(self):
        cds = self._pdm.recompute_conditional_structure_structure(self._best_meshes_mni[0].gen_num_of_points())
        for i_test_sub in range(len(self._test_subj)):



            for lab in range(len(settings.settings.dependent_constraint.keys())):
                fc = FunctionHandlerMulti()
                #todo fix set
                fc.set_image(  self._test_subj[i_test_sub] + os.sep + "t2_acpc_normalised.nii.gz/t2_resampled_fcm.nii.gz"   )
                fc.set_subject(self._test_subj[i_test_sub])
                fc._mesh = self._best_meshes_mni[0]
                fc._cmesh = self._best_meshes_c[0]

                keys1 = list(settings.settings.dependent_constraint.keys())[lab]

                keys2= keys1.split(',')

                num_of_pts = None
                joined_s2 = None
                for i in range(len(keys2)):

                    mesh = ExtPy.cMesh(self._test_subj[i_test_sub] + os.sep + keys2[i] + "_fitted.obj")
                    a = mesh.get_unpacked_coords()
                    num_of_pts = len(a)/3
                    if joined_s2 is None:
                        joined_s2 = np.array(a)
                    else:
                        joined_s2 = np.concatenate(joined_s2,a)


                fc._mesh.calculate_closest_points()
                fc._kdes = cds[lab]
                fc._kdes.set_S2(joined_s2) # set readed shape

                fc._num_of_points= int(num_of_pts)

                #constraint for interception
                vals1 = settings.settings.dependent_constraint[keys1].split(',')

                fc._constraints = len(vals1)

                X0 = np.zeros((fc._kdes.get_num_eigenvecs()))

                #####SAVE initi
                res_points = fc._kdes.vector_2_points(X0)
                #cmp =

                for i in range(len(vals1 )):

                    Xr = res_points[3*fc._num_of_points*i:3*fc._num_of_points*(i+1)]

                    l = len(Xr.tolist())
                    fc._mesh.modify_points(Xr.reshape(( int(l/3),3)))
                    fc._mesh.apply_transform(utils.read_fsl_mni2native_w(self._test_subj[i_test_sub]))
                    fc._mesh.save_obj(self._test_subj[i_test_sub] + os.sep + vals1[i] + "_initialise.obj")



                ##########################################################
                print(datetime.now())
                #print(fc._cmesh.selfIntersectionTest(list(cds[lab][0].vector_2_points(X0))))
                print(datetime.now())
                #print(fc._mesh.calculate_interception_from_newPTS(np.array(X0)))
                print(datetime.now())


                bounds = fc._kdes.generate_bounds(3)


                print("Start optimising ")
                mimiser = opt.minimize(fc, X0, method='TNC',bounds=bounds, options={"disp": True})
                r_x = mimiser.fun
                while True:
                    mimiser = opt.minimize(fc, mimiser.x, method='TNC', bounds=bounds,tol=0.5, options={"disp": True, 'ftol': 0.5})
                    mimiser = opt.minimize(fc, mimiser.x, method='Powell', bounds=bounds,
                                          options={"disp": True
                                               ,'ftol':0.5
                                                   })

                    if r_x - mimiser.fun < 0.1:
                        break
                    r_x = mimiser.fun
                #mimiser = opt.minimize(fc, X0, method='L-BFGS-B', bounds=bounds, options={"disp": True})
                #mimiser = opt.minimize(fc, X0, method="cg", options={"disp": True})

                #mimiser = opt.minimize(fc, X0,method='COBYLA',tol=1,constraints=con,options={"maxiter":5000})
                print(datetime.now())
                res_points = fc._kdes.vector_2_points(mimiser.x)

                for i in range(len(vals1 )):

                    Xr = res_points[3*fc._num_of_points*i:3*fc._num_of_points*(i+1)]


                    l = len(Xr.tolist())
                    fc._mesh.modify_points(Xr.reshape(( int(l/3),3)))
                    fc._mesh.apply_transform(utils.read_fsl_mni2native_w(self._test_subj[i_test_sub]))

                    fc._mesh.save_obj(self._test_subj[i_test_sub] + os.sep + vals1[i] + "_fitted.obj")

                a = fc(X0)
            pass
        pass


        pass