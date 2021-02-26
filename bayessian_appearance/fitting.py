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

import threading

import sys

import ExtPy

import bayessian_appearance.settings as gl_set

import scipy.optimize as opt


def save_normal_as_csv(normals, subj):
    try:
        os.remove(path=subj + os.sep + "normals.fcsv")
    except:
        pass

    with open(subj + os.sep + "normals.fcsv", 'wt') as the_file:
        the_file.write("# Markups fiducial file version = 4.10\n")
        the_file.write("# CoordinateSystem = 0\n")
        the_file.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")

        for i in range(len(normals)):
            st = str(i) + ","
            st += str(normals[i][0]) + "," + str(normals[i][1]) + "," + str(normals[i][2])

            st += ",0.000,0.000,0.000,1.000,1,1,0,n " + str(i) + ",,vtkMRMLScalarVolumeNode1\n"
            the_file.write(st)


def compute_pd_single_labed(subject, label, func, pca):
    cm = ExtPy.cMesh(subject + os.sep + label + "_1.obj")

    cdw = cm.get_unpacked_coords()

    norm2 = cm.generate_normals(gl_set.settings.norm_length, gl_set.settings.discretisation)
    save_normal_as_csv(norm2[0], subject)

    cm.apply_transform(np.linalg.inv(func._mni_world).tolist())
    coords = cm.get_unpacked_coords()

    coords = np.array(coords)
    coords = coords.reshape((1, coords.shape[0]))
    coords = pca[0].transform(coords)[0]
    coords = func._kdes[0].decompose_coords_to_eigcords(coords)

    return func(coords)


# for single mesh
class FunctionHandler:
    _image = None
    _cimage = None
    _from_mni_to_vox = None
    _vox_to_world = None
    # label of single mesh
    _label = None
    _mesh = None
    _cmesh = None
    _num_of_points = None
    _mni_world = None

    intens_pca = None
    shape_pca = None
    mean_intens_distr = None
    volume_distr = None
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

    # TODO rewrite to make it universal
    def set_subject(self, subject_dir):
        mni_im = fim.Image(subject_dir + os.sep + "t1_brain_to_mni_stage2_apply.nii.gz")

        native_im = fim.Image(subject_dir + os.sep + "t1_acpc_extracted.nii.gz")

        from_mni_mat = fl.readFlirt(subject_dir + os.sep + "combined_affine_reverse.mat")

        mni_w = fl.fromFlirt(from_mni_mat, mni_im, native_im, "world", "world")
        self._mni_world = mni_w
        a = np.linalg.inv(self._vox_to_world)
        self._from_mni_to_vox = np.dot(a, mni_w)

    def __call__(self, *args, **kwargs):
        coords = np.array(args[0])

        coords = self._kdes[0].vector_2_points(coords)

        coords1 = self.shape_pca.inverse_transform(coords.reshape((1, coords.shape[0])))[0]

        self._cmesh.modify_points(coords1)
        normals = self._cmesh.generate_normals(gl_set.settings.norm_length, gl_set.settings.discretisation)

        mesh_pts = ExtPy.apply_transform_2_pts(self._cmesh.generate_mesh_points(10), self._from_mni_to_vox.tolist())
        # if not mesh_pts:
        #     utils.points_2_fcsv(self._cmesh.get_mesh_boundary_roi(10),"/home/varga/processing_data/workdir/mesh.fcsv")
        #     a = self._cmesh.is_points_inside(self._cmesh.get_mesh_boundary_roi(10))
        #     self._cmesh.save_obj("/home/varga/processing_data/workdir/31.obj")
        #     mesh_pts = ExtPy.apply_transform_2_pts(self._cmesh.generate_mesh_points(30), self._from_mni_to_vox.tolist())

        normals = ExtPy.apply_transform_2_norms(normals, self._from_mni_to_vox.tolist())

        # intens
        ips = np.nanmean(np.array(self._image.interpolate_list(mesh_pts)))

        norm_intensities = np.array(self._image.interpolate_normals(normals))

        # normalise intensity
        norm_intensities = norm_intensities - ips
        mes_vol = self._cmesh.calculate_volume()
        norm_intensities = norm_intensities.reshape((1, norm_intensities.shape[0] * norm_intensities.shape[1]))

        norm_intensities = self.intens_pca.transform(norm_intensities)[0]
        # coords[:] = 0 # debug
        # norm_intens[:] = 0 #for debug
        # distr_coords = np.concatenate((coords,norm_intens))
        return (- self._kdes[1](coords, norm_intensities)
                + self.mean_intens_distr.mahalanobis(ips.reshape(1, 1))[0]
                # + self.volume_distr.mahalanobis(np.array([mes_vol]).reshape(1,1))[0]
                )

    def set_cimage(self, filename):
        vt = ExtPy.cImage(filename)
        self._cimage = vt
        pass


# for multi mesh
class FunctionHandlerMulti:
    _image = None
    _from_mni_to_vox = None
    _vox_to_world = None
    # label of single mesh
    _label = None
    _mesh = None
    _cmesh = None
    _num_of_points = None
    _mni_world = None

    intens_pcas = None
    shape_pcas = None
    shape_lens = None

    mean_intens_distr = None

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

    # TODO rewrite to make it universal
    def set_subject(self, subject_dir):

        mni_im = fim.Image(subject_dir + os.sep + "t1_brain_to_mni_stage2_apply.nii.gz")

        native_im = fim.Image(subject_dir + os.sep + "t1_acpc_extracted.nii.gz")

        from_mni_mat = fl.readFlirt(subject_dir + os.sep + "combined_affine_reverse.mat")

        mni_w = fl.fromFlirt(from_mni_mat, mni_im, native_im, "world", "world")
        self._mni_world = mni_w
        a = np.linalg.inv(self._vox_to_world)
        self._from_mni_to_vox = np.dot(a, mni_w)

    def __call__(self, *args, **kwargs):
        coords = np.array(args[0])

        coords = self._kdes.vector_2_points(coords)

        nits = None
        mah = 0
        for i in range(self._constraints):

            st = sum(self.shape_lens[:i])
            cd = coords[st:st + self.shape_lens[i]].reshape((1, self.shape_lens[i]))
            cd = self.shape_pcas[i].inverse_transform(cd)[0]
            self._cmesh.modify_points(cd)

            normals = self._cmesh.generate_normals(gl_set.settings.norm_length, gl_set.settings.discretisation)
            # print(datetime.now())
            mesh_pts = ExtPy.apply_transform_2_pts(self._cmesh.generate_mesh_points(10), self._from_mni_to_vox.tolist())

            normals = ExtPy.apply_transform_2_norms(normals, self._from_mni_to_vox)

            ###here are mean intensity normalisation( removed )
            ips = np.nanmean(np.array(self._image.interpolate_list(mesh_pts)))
            mah += self.mean_intens_distr[i].mahalanobis(ips.reshape(1, 1))[0]

            norm_intens = np.array(self._image.interpolate_normals(normals))

            # normalise intensity
            # ips = 0  # need to remove aftr
            norm_intens = norm_intens - ips

            norm_intens = norm_intens.reshape((1, norm_intens.shape[0] * norm_intens.shape[1]))
            norm_intens = self.intens_pcas[i].transform(norm_intens)[0]
            if nits is None:
                nits = norm_intens
            else:
                nits = np.concatenate((nits, norm_intens))
        # distr_coords = np.concatenate((coords,norm_intens))

        return (- self._kdes(coords, nits)) + mah


class FunctionHandlerJoint:
    _image = None
    _from_mni_to_vox = None
    _vox_to_world = None
    # label of single mesh
    _label = None
    _mesh = None
    _cmesh = None
    _num_of_points = None
    _mni_world = None

    intens_pcas = None
    shape_pcas = None
    shape_lens = None

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

    # TODO rewrite to make it universal
    def set_subject(self, subject_dir):

        mni_im = fim.Image(subject_dir + os.sep + "t1_brain_to_mni_stage2_apply.nii.gz")

        native_im = fim.Image(subject_dir + os.sep + "t1_acpc_extracted.nii.gz")

        from_mni_mat = fl.readFlirt(subject_dir + os.sep + "combined_affine_reverse.mat")

        mni_w = fl.fromFlirt(from_mni_mat, mni_im, native_im, "world", "world")
        self._mni_world = mni_w
        a = np.linalg.inv(self._vox_to_world)
        self._from_mni_to_vox = np.dot(a, mni_w)

    def __call__(self, *args, **kwargs):
        coords = np.array(args[0])

        coords = self._kdes.vector_2_points(coords)

        nits = None
        for i in range(self._constraints):

            st = sum(self.shape_lens[:i])
            cd = coords[st:st + self.shape_lens[i]].reshape((1, self.shape_lens[i]))
            cd = self.shape_pcas[i].inverse_transform(cd)[0]
            self._cmesh.modify_points(cd)

            normals = self._cmesh.generate_normals(gl_set.settings.norm_length, gl_set.settings.discretisation)
            # print(datetime.now())
            # mesh_pts = ExtPy.apply_transform_2_pts(self._cmesh.generate_mesh_points(10),self._from_mni_to_vox.tolist())

            normals = ExtPy.apply_transform_2_norms(normals, self._from_mni_to_vox)

            ###here are mean intensity normalisation( removed )
            # ips = np.nanmean(np.array(self._image.interpolate_list(mesh_pts)))

            norm_intens = np.array(self._image.interpolate_normals(normals))

            # normalise intensity
            ips = 0  # need to remove aftr
            norm_intens = norm_intens - ips

            norm_intens = norm_intens.reshape((1, norm_intens.shape[0] * norm_intens.shape[1]))
            norm_intens = self.intens_pcas[i].transform(norm_intens)[0]
            if nits is None:
                nits = norm_intens
            else:
                nits = np.concatenate((nits, norm_intens))
        # distr_coords = np.concatenate((coords,norm_intens))

        return - self._kdes(coords, nits)


class Fitter:
    _pdm = None

    _train_subj = None
    _test_subj = None
    _modal = None
    _mat_over = None
    _best_meshes_mni = None
    _best_meshes_c = None

    def __init__(self, train_subj, test_subj, pdm=None):
        self._train_subj = train_subj
        self._test_subj = test_subj
        if pdm != None:
            if isinstance(pdm, str):
                self._pdm = pd.PointDistribution.read_pdm(file_name=pdm)

            else:
                self._pdm = pdm

    def read_pdm(self, filename):
        self._pdm = pd.PointDistribution.read_pdm(file_name=filename)

    def set_modalities(self, modalities):
        self._modal = modalities

    def set_overlaped(self, file_name):
        mat = np.loadtxt(file_name)
        self._mat_over = mat

        # get best meshes of segmented figures
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
            # surf= mesh.Mesh(self._train_subj[sub_ind] + os.sep + label + "_1.obj")
            surf_c = ExtPy.cMesh(gl_set.settings.atlas_dir + os.sep + label + "m.obj")
            surf = mesh.Mesh(gl_set.settings.atlas_dir + os.sep + label + "m.obj")
            # surf.apply_transform(mni_w)
            # surf_c.applyTransformation(mni_w.tolist())
            self._best_meshes_c.append(surf_c)
            self._best_meshes_mni.append(surf)

    def _fit_single(self, X0, fc, bounds, ind_of_iter, subj, label):
        try:
            X0 = fc._kdes[0].decompose_coords_to_eigcords(X0)
            #####SAVE initi
            res_points = fc._kdes[0].vector_2_points(X0)
            res_points = fc.shape_pca.inverse_transform(res_points.reshape((1, res_points.shape[0])))[0]
            l = len(res_points.tolist())
            fc._mesh.modify_points(res_points.reshape((int(l / 3), 3)))
            fc._mesh.apply_transform(utils.read_fsl_mni2native_w(subj))
            fc._mesh.save_obj(subj + os.sep + label + str(ind_of_iter) + "_initialise.obj")

            print("Start optimising ")
            # mimiser = opt.minimize(fc, X0, method='TNC', bounds=bounds, options={"disp": True})
            mimiser = opt.minimize(fc, X0, method='Powell', bounds=bounds,
                                   options={"disp": True,
                                            # 'ftol': 0.5
                                            })
            r_x = mimiser.fun
            mimiser_t = mimiser
            while True:
                # mimiser = opt.minimize(fc, mimiser.x, method='TNC', bounds=bounds, options={"disp": True
                #                                                                              , 'ftol': 0.5
                #                                                                             })
                mimiser = opt.minimize(fc, mimiser.x, method='Powell', bounds=bounds,
                                       options={"disp": True,
                                                # 'ftol':0.5,
                                                # 'direc' : mimiser.direc
                                                })

                if ((r_x - mimiser.fun) < 0.5) and ((r_x - mimiser.fun) > 0):
                    break
                elif (r_x - mimiser.fun) < 0:
                    mimiser = mimiser_t
                    break
                r_x = mimiser.fun
                mimiser_t = mimiser
            return mimiser
        except Exception as e:
            print(e)
            return None
            pass

    def fit_single(self):
        cds = self._pdm.recompute_conditional_shape_int_distribution(self._best_meshes_mni[0].gen_num_of_points())

        for i_test_sub in range(len(self._test_subj)):
            print("###################################################")
            print("Subject")
            print(self._test_subj[i_test_sub])
            print("---------------------------------------")
            for lab in range(len(self._best_meshes_mni)):
                print("LABEL")
                print(self._pdm._label_kde[lab])
                ind_of_el = utils.comp_posit_in_data(self._pdm._label_kde[lab])  # across all labels
                fc = FunctionHandler()
                # todo fix set
                fc.set_image(self._test_subj[i_test_sub] + os.sep + "t2_acpc_normalised.nii.gz/t2_resampled_fcm.nii.gz")
                # fc.set_cimage(self._test_subj[i_test_sub] + os.sep + "t2_acpc_normalised.nii.gz/t2_resampled_fcm.nii.gz")
                fc.set_subject(self._test_subj[i_test_sub])
                fc._mesh = self._best_meshes_mni[lab]
                fc._cmesh = self._best_meshes_c[lab]
                fc.volume_distr = self._pdm.pdfs_vol[ind_of_el]
                fc.mean_intens_distr = self._pdm.pdfs_int[ind_of_el]
                # copute pca
                fc.shape_pca = self._pdm.get_shape_pca(self._pdm._label_kde[lab])
                fc.intens_pca = self._pdm.get_intens_pca(self._pdm._label_kde[lab])
                fc._mesh.calculate_closest_points()
                fc._kdes = cds[lab]
                fc._num_of_points = self._best_meshes_mni[lab].gen_num_of_points()

                # constraint for interception
                # int_Cons = opt.NonlinearConstraint(fc._mesh.calculate_interception_from_newPTS,lb=0,ub=2)

                X0 = fc._mesh.get_unpacked_coords()
                X0 = fc.shape_pca.transform(np.array(X0).reshape((1, len(X0))))[0]
                # X0 = cds[lab][0].decompose_coords_to_eigcords(X0)

                # X0[:] = 0
                X0 = cds[lab][0]._pca.transform(cds[lab][2][0, :, :])[0]
                ##########################################################
                print(datetime.now())
                Xx = cds[lab][0].vector_2_points(X0)
                print(fc._cmesh.selfIntersectionTest(
                    fc.shape_pca.inverse_transform(np.array(Xx).reshape((1, len(Xx))))[0]))
                print(datetime.now())
                print(datetime.now())

                a = fc(X0)
                print(" INITIALISE AT MEDIAN ")
                print(a)
                print("------------------------------------------------------")

                b = compute_pd_single_labed(self._test_subj[i_test_sub], self._pdm._label_kde[lab], fc,
                                            [fc.shape_pca, fc.intens_pca])

                print(" FUNC VALUE AT MANUAL")
                print(b)
                print("------------------------------------------------------")
                bounds = cds[lab][0].generate_bounds(3)

                components = self._pdm.compute_4_median_components(label=self._pdm._label_kde[lab], distr=fc._kdes[1])
                values = []
                it = 0
                for comp in components:
                    values.append(self._fit_single(comp, fc, bounds=bounds, ind_of_iter=it
                                                   , subj=self._test_subj[i_test_sub], label=self._pdm._label_kde[lab]))
                    it += 1
                values = [v for v in values if v]
                a = min(values, key=lambda k: k.fun)
                ind = 0
                for ind1 in range(len(values)):
                    if values[ind1] is None:
                        continue
                    if values[ind1].fun == a.fun:
                        ind = ind1
                        break

                r_x = values[ind].fun
                mimiser = values[ind]
                print(" FUNC VALUE AT FINAL")
                print(r_x)
                print("------------------------------------------------------")

                res_points = cds[lab][0].vector_2_points(mimiser.x)
                res_points = fc.shape_pca.inverse_transform(res_points.reshape((1, res_points.shape[0])))[0]
                l = len(res_points.tolist())
                fc._mesh.modify_points(res_points.reshape((int(l / 3), 3)))
                fc._mesh.apply_transform(utils.read_fsl_mni2native_w(self._test_subj[i_test_sub]))

                fc._mesh.save_obj(self._test_subj[i_test_sub] + os.sep + self._pdm._label_kde[lab] + "_fitted.obj")
                print(" END SUBJECT")
                print("------------------------------------------------------")
            pass
        pass


    def _fit_multi(self,X0, fc, bounds, ind_of_iter, subj, labels ):

        try:
            X0 = fc._kdes.decompose_coords_to_eigcords(X0)
            res_points = fc._kdes.vector_2_points(X0)
            # cmp =

            shape_lens = []  # len of shapes
            shape_pcas = []
            intens_pcas = []
            fc.volume_distr = []
            fc.mean_intens_distr = []
            for i in range(len(labels)):
                ind_of_label = utils.comp_posit_in_data(labels[i])
                coords1 = self._pdm.get_shape_coords(ind_of_label)
                len1 = coords1[1] - coords1[0]
                shape_lens.append(len1)
                st = shape_lens[:i]
                st = sum(st)
                Xr = res_points[st:st + len1]
                # o
                fc.volume_distr.append(self._pdm.pdfs_vol[ind_of_label])
                fc.mean_intens_distr.append(self._pdm.pdfs_int[ind_of_label])

                pca1 = self._pdm.get_shape_pca(labels[i])
                shape_pcas.append(pca1)  # shape pcas
                intens_pcas.append(self._pdm.get_intens_pca(labels[i]))
                Xr = pca1.inverse_transform(Xr.reshape((1, Xr.shape[0])))[0]
                l = len(Xr.tolist())
                fc._mesh.modify_points(Xr.reshape((int(l / 3), 3)))
                fc._mesh.apply_transform(utils.read_fsl_mni2native_w(subj))
                fc._mesh.save_obj(subj + os.sep + labels[i] + str(ind_of_iter)+"_initialise.obj")

            fc.shape_lens = shape_lens
            fc.intens_pcas = intens_pcas
            fc.shape_pcas = shape_pcas

            ##########################################################
            print(datetime.now())
            # print(fc._cmesh.selfIntersectionTest(list(cds[lab][0].vector_2_points(X0))))
            print(datetime.now())
            # print(fc._mesh.calculate_interception_from_newPTS(np.array(X0)))
            print(datetime.now())



            print("Start optimising ")
            mimiser = opt.minimize(fc, X0, method='Powell', bounds=bounds,
                                   options={"disp": True
                                            #                                        ,'ftol':0.5
                                            })

            r_x = mimiser.fun
            mimiser_t = mimiser
            while True:
                # mimiser = opt.minimize(fc, mimiser.x, method='TNC', bounds=bounds, options={"disp": True,
                #                                                                             # 'ftol': 0.5
                #                                                                             })
                mimiser = opt.minimize(fc, mimiser.x, method='Powell', bounds=bounds,
                                       options={"disp": True
                                                #                                        ,'ftol':0.5
                                                })

                if ((r_x - mimiser.fun) < 0.1) and ((r_x - mimiser.fun) > 0):
                    break
                elif (r_x - mimiser.fun) < 0:
                    mimiser = mimiser_t
                    break
                mimiser_t = mimiser
                r_x = mimiser.fun
            # mimiser = opt.minimize(fc, X0, method='L-BFGS-B', bounds=bounds, options={"disp": True})
            # mimiser = opt.minimize(fc, X0, method="cg", options={"disp": True})

            # mimiser = opt.minimize(fc, X0,method='COBYLA',tol=1,constraints=con,options={"maxiter":5000})
            print(datetime.now())
            return mimiser
        except Exception as e:
            print(e)
            return None



    def fit_multiple(self):
        if not settings.settings.joint_labels:
            return
        cds = self._pdm.compute_joined_structures_model()
        for i_test_sub in range(len(self._test_subj)):
            print("###################################################")
            print("Subject")
            print(self._test_subj[i_test_sub])
            print("---------------------------------------")
            for lab in range(len(settings.settings.joint_labels)):

                fc = FunctionHandlerMulti()
                # todo fix set
                fc.set_image(self._test_subj[i_test_sub] + os.sep + "t2_acpc_normalised.nii.gz/t2_resampled_fcm.nii.gz")
                fc.set_subject(self._test_subj[i_test_sub])
                fc._mesh = self._best_meshes_mni[0]
                fc._cmesh = self._best_meshes_c[0]

                # keys1 = list(settings.settings.dependent_constraint.keys())[lab]
                #
                keys2 = settings.settings.joint_labels[lab]

                num_of_pts = None
                joined_s2 = None

                fc._mesh.calculate_closest_points()
                fc._kdes = cds[lab]

                fc._num_of_points = int(0)

                # constraint for interception

                fc._constraints = len(keys2)
                bounds = fc._kdes.generate_bounds(2.5)
                shape_pcas = []
                shape_lens = []
                for p_lab in range(len(keys2)):
                    ind_of_label = utils.comp_posit_in_data(keys2[p_lab])
                    coords1 = self._pdm.get_shape_coords(ind_of_label)
                    len1 = coords1[1] - coords1[0]
                    shape_lens.append(len1)
                    pca1 = self._pdm.get_shape_pca(keys2[p_lab])
                    shape_pcas.append(pca1)  # shape pcas
                # X0 = np.zeros((fc._kdes.get_num_eigenvecs()))

                #####SAVE initi

                Xs = fc._kdes.compute_median(3,4)
                x_i = 0
                res_mimisers = []
                it = 0
                for comp in Xs:
                    res_mimisers.append(self._fit_multi(comp, fc=fc, bounds=bounds, ind_of_iter=it
                                                   , subj=self._test_subj[i_test_sub], labels=keys2))
                    it += 1
                values = [v for v in res_mimisers if v]
                a = min(values, key=lambda k: k.fun)
                ind = 0
                for ind1 in range(len(values)):
                    if values[ind1] is None:
                        continue
                    if values[ind1].fun == a.fun:
                        ind = ind1
                        break

                r_x = values[ind].fun
                mimiser = values[ind]


                res_points = fc._kdes.vector_2_points(mimiser.x)
                for i in range(len(keys2)):
                    st = shape_lens[:i]
                    st = sum(st)
                    Xr = res_points[st:st + shape_lens[i]]

                    Xr = shape_pcas[i].inverse_transform(Xr.reshape((1, Xr.shape[0])))[0]
                    l = len(Xr.tolist())
                    fc._mesh.modify_points(Xr.reshape((int(l / 3), 3)))
                    fc._mesh.apply_transform(utils.read_fsl_mni2native_w(self._test_subj[i_test_sub]))

                    fc._mesh.save_obj(self._test_subj[i_test_sub] + os.sep + keys2[i] + "_fitted.obj")

            print("###################END SUB#######################")
            print("Subject")
            print(self._test_subj[i_test_sub])
            print("#################################################")
            pass

        pass

        pass
# OLD
# def fit_multiple(self):
#     if not settings.settings.dependent_constraint:
#         return
#     cds = self._pdm.recompute_conditional_structure_structure(self._best_meshes_mni[0].gen_num_of_points())
#     for i_test_sub in range(len(self._test_subj)):
#
#         for lab in range(len(settings.settings.dependent_constraint.keys())):
#             fc = FunctionHandlerMulti()
#             # todo fix set
#             fc.set_image(self._test_subj[i_test_sub] + os.sep + "t2_acpc_normalised.nii.gz/t2_resampled_fcm.nii.gz")
#             fc.set_subject(self._test_subj[i_test_sub])
#             fc._mesh = self._best_meshes_mni[0]
#             fc._cmesh = self._best_meshes_c[0]
#
#             keys1 = list(settings.settings.dependent_constraint.keys())[lab]
#
#             keys2 = keys1.split(',')
#
#             num_of_pts = None
#             joined_s2 = None
#             for i in range(len(keys2)):
#
#                 mesh = ExtPy.cMesh(self._test_subj[i_test_sub] + os.sep + keys2[i] + "_fitted.obj")
#                 a = mesh.get_unpacked_coords()
#                 ind_key = utils.comp_posit_in_data(keys2[i])  # index of label
#                 num_of_pts = len(a) / 3
#                 a = self._pdm.get_shape_pca(ind_key).transform(np.array(a).reshape((1, len(a))))[0]
#                 if joined_s2 is None:
#                     joined_s2 = np.array(a)
#                 else:
#                     joined_s2 = np.concatenate(joined_s2, a)
#
#             fc._mesh.calculate_closest_points()
#             fc._kdes = cds[lab]
#             fc._kdes.set_S2(joined_s2)  # set readed shape
#             X0 = fc._kdes.get_mean_conditional()
#             X0 = fc._kdes.decompose_coords_to_eigcords(X0)
#             fc._num_of_points = int(num_of_pts)
#
#             # constraint for interception
#             vals1 = settings.settings.dependent_constraint[keys1].split(',')
#
#             fc._constraints = len(vals1)
#
#             # X0 = np.zeros((fc._kdes.get_num_eigenvecs()))
#
#             #####SAVE initi
#             res_points = fc._kdes.vector_2_points(X0)
#             # cmp =
#
#             shape_lens = []  # len of shapes
#             shape_pcas = []
#             intens_pcas = []
#             for i in range(len(vals1)):
#                 ind_of_label = utils.comp_posit_in_data(vals1[i])
#                 coords1 = self._pdm.get_shape_coords(ind_of_label)
#                 len1 = coords1[1] - coords1[0]
#                 shape_lens.append(len1)
#                 st = shape_lens[:i]
#                 st = sum(st)
#                 Xr = res_points[st:st + len1]
#
#                 pca1 = self._pdm.get_shape_pca(ind_of_label)
#                 shape_pcas.append(pca1)  # shape pcas
#                 intens_pcas.append(self._pdm.get_intens_pca(ind_of_label))
#                 Xr = pca1.inverse_transform(Xr.reshape((1, Xr.shape[0])))[0]
#                 l = len(Xr.tolist())
#                 fc._mesh.modify_points(Xr.reshape((int(l / 3), 3)))
#                 fc._mesh.apply_transform(utils.read_fsl_mni2native_w(self._test_subj[i_test_sub]))
#                 fc._mesh.save_obj(self._test_subj[i_test_sub] + os.sep + vals1[i] + "_initialise.obj")
#
#             fc.shape_lens = shape_lens
#             fc.intens_pcas = intens_pcas
#             fc.shape_pcas = shape_pcas
#
#             ##########################################################
#             print(datetime.now())
#             # print(fc._cmesh.selfIntersectionTest(list(cds[lab][0].vector_2_points(X0))))
#             print(datetime.now())
#             # print(fc._mesh.calculate_interception_from_newPTS(np.array(X0)))
#             print(datetime.now())
#
#             bounds = fc._kdes.generate_bounds(3)
#
#             print("Start optimising ")
#             mimiser = opt.minimize(fc, X0, method='TNC', bounds=bounds, options={"disp": True})
#             r_x = mimiser.fun
#             while True:
#                 mimiser = opt.minimize(fc, mimiser.x, method='TNC', bounds=bounds, options={"disp": True,
#                                                                                             # 'ftol': 0.5
#                                                                                             })
#                 mimiser = opt.minimize(fc, mimiser.x, method='Powell', bounds=bounds,
#                                        options={"disp": True
#                                                 #                                        ,'ftol':0.5
#                                                 })
#
#                 if r_x - mimiser.fun < 0.1:
#                     break
#                 r_x = mimiser.fun
#             # mimiser = opt.minimize(fc, X0, method='L-BFGS-B', bounds=bounds, options={"disp": True})
#             # mimiser = opt.minimize(fc, X0, method="cg", options={"disp": True})
#
#             # mimiser = opt.minimize(fc, X0,method='COBYLA',tol=1,constraints=con,options={"maxiter":5000})
#             print(datetime.now())
#             res_points = fc._kdes.vector_2_points(mimiser.x)
#
#             for i in range(len(vals1)):
#                 st = shape_lens[:i]
#                 st = sum(st)
#                 Xr = res_points[st:st + shape_lens[i]]
#
#                 Xr = shape_pcas[i].inverse_transform(Xr.reshape((1, Xr.shape[0])))[0]
#                 l = len(Xr.tolist())
#                 fc._mesh.modify_points(Xr.reshape((int(l / 3), 3)))
#                 fc._mesh.apply_transform(utils.read_fsl_mni2native_w(self._test_subj[i_test_sub]))
#
#                 fc._mesh.save_obj(self._test_subj[i_test_sub] + os.sep + vals1[i] + "_fitted.obj")
#
#             a = fc(X0)
#         pass
#     pass
#
#     pass
