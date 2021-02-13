import bayessian_appearance.settings as settings
import bayessian_appearance.distributions as distros
import os
import csv
import sklearn.neighbors as neighb
import sklearn.decomposition as decomp
import distutils.util
import numpy as np
import pickle
import ExtPy
import bayessian_appearance.utils as utils

import scipy.stats as scp_stats


class PointDistribution:
    use_constraint = None
    _original_data = None
    _tr_subjects =None
    _num_of_pts = None

    _median_all = None
    _labels = None

    _shape_coords = None
    _intens_coords = None
    #shape+intens data
    shape_data = None
    intens_data = None

    shape_pca = None
    intens_pca = None

    _kdes = None
    # description of each label
    _label_kde = None

    # extrapolate to false
    def _prolongue_label(self, label_vec, mask_vec):
        res = label_vec.copy()
        i_f = mask_vec.index(True)
        # if min(mask_vec) == False:
        #     print("HAS FALSE")
        ##f
        for i in (range(i_f)):
            res[i] = res[i_f]
        return res



    def _compute_insities(self, labels, subjects):
        #recompute subjects
        for subj in subjects:

            utils.calculate_intensites_subject(modalities=settings.settings.modalities,labels=labels,subject=subj
                                               ,discretisation=settings.settings.discretisation,norm_len=settings.settings.norm_length,
                                               mesh_name_end="_pca.obj")


        self._original_data = self._read_labels(labels=labels,train_subjects=subjects)
        new_intens = []
        pcas = []
        for lab_i in range(len(labels)):
            pdm_label_data = self._original_data[lab_i]

            intensity_profs = []
            intensity_kdes = []
            for vert in range(len(pdm_label_data[0])):
                profile = []
                for sub in range(len(pdm_label_data)):
                    single_profile = self._prolongue_label(label_vec=pdm_label_data[sub][vert][2],
                                                           mask_vec=pdm_label_data[sub][vert][1])
                    profile.append(single_profile)

                intensity_profs.append(profile)
                # i_kde = scp_stats.gaussian_kde(profile.transpose()) #TODO MAYBE ADD CONSTRAINTS FOR OUTSIDE

            # here we form joint intensity rows = sample col = coords (x y z)
            np_intensties = None
            for j in range(len(pdm_label_data[0])):
                if j == 0:
                    np_intensties = np.array(intensity_profs[j])
                    continue
                np_intensties = np.concatenate((np_intensties, np.array(intensity_profs[j])), axis=1)

            #compute PCA
            pca = decomp.PCA(n_components=settings.settings.pca_precision,svd_solver='full')
            pca.fit(np_intensties)
            pcas.append(pca)
            #compute
            np_intensties_new = pca.transform(np_intensties)
            new_intens.append(np_intensties_new)
        self._pca_intensities = pcas
        return new_intens


        pass
    def _compute_pca_components(self, labels, subjects,datasets ):

        pca_shapes = []

        new_dataset = [] # those new data
        for lab_i in range(len(labels)):

            data = datasets[lab_i]
            #compute pca
            pca = decomp.PCA(n_components=settings.settings.pca_precision,svd_solver='full')
            pca.fit(data)
            pca_shapes.append(pca)
            # recompute shapes
            new_shape_coords = None
            for sub_i in range(len(subjects)):

                el = ExtPy.cMesh(subjects[sub_i] + os.sep + labels[lab_i] + "_1.obj")

                points = el.get_unpacked_coords()


                to_mni = utils.read_fsl_native2mni_w(subjects[sub_i])
                el.apply_transform(to_mni)
                points = el.get_unpacked_coords()
                pts  = pca.inverse_transform(pca.transform([points]))
                if new_shape_coords is None:
                    new_shape_coords = pts[0]
                else:
                    new_shape_coords = np.concatenate((new_shape_coords,pts[0]))
                el.modify_points(pts[0])
                el.apply_transform(np.linalg.inv(to_mni))
                el.save_obj(subjects[sub_i] + os.sep + labels[lab_i] + "_pca.obj")


            new_dataset.append(pca.transform(data))


        self._pca_shapes = pca_shapes
        return new_dataset
        pass

    def _exclude_norm(self):
        pass

    def _construct_pdm(self):

        # find what we segmenting
        list_pos = []  # positions of labels which we segmenting
        for i in range(len(settings.settings.labels_to_segment)):
            pos = self._labels.index(settings.settings.labels_to_segment[i])
            list_pos.append(pos)

        pass

        posit_of_cent = int(settings.settings.discretisation / 2)

        kde_combined = []
        # construct combined ONE PDM for ALL labels here
        comb_array = None


        comb_shapes = [] # array of array of shapes
        for i in range(len(self._labels)):
            pdm_label_data = self._original_data[i]
            # get locations]

            # calculate cumulative coordinate of each coord. row sample col
            coordinates = []
            for j in range(len(pdm_label_data[0])):
                vec_j = [x[j][0][posit_of_cent] for x in pdm_label_data]
                coordinates.append(vec_j)

            # generate kdes for coords
            posit_kdes = []

            # here we form joint coordinates rows = sample col = coords (x y z)
            np_coord = None
            for j in range(len(coordinates)):
                if j == 0:
                    np_coord = np.array(coordinates[j])
                    continue
                np_coord = np.concatenate((np_coord, np.array(coordinates[j])), axis=1)

            #add coordinates
            comb_shapes.append(np_coord)
            #############INTENSITY KDES
            ##RECord KDE like as whole profile (ESTIMATE THROUGH Whoole profile)!!!
            kdes_norms = []
            # TODO add multiple modalities here
            # num_of_mods = int(len(pdm_label_data[0][0][2]) / settings.settings.discretisation)
            #
            # intensity_profs = []
            # for vert in range(len(pdm_label_data[0])):
            #     profile = []
            #     for sub in range(len(pdm_label_data)):
            #         single_profile = self._prolongue_label(label_vec=pdm_label_data[sub][vert][2],
            #                                                mask_vec=pdm_label_data[sub][vert][1])
            #         profile.append(single_profile)
            #
            #     intensity_profs.append(profile)
            #     # i_kde = scp_stats.gaussian_kde(profile.transpose()) #TODO MAYBE ADD CONSTRAINTS FOR OUTSIDE
            #
            # # here we form joint intensity rows = sample col = coords (x y z)
            # np_intensties = None
            # for j in range(len(pdm_label_data[0])):
            #     if j == 0:
            #         np_intensties = np.array(intensity_profs[j])
            #         continue
            #     np_intensties = np.concatenate((np_intensties, np.array(intensity_profs[j])), axis=1)

            # formulate shape+intensity distribution
            # if comb_array is None:
            #
            #     comb_array = np.concatenate((np_coord, np_intensties), axis=1)
            # else:
            #     a = np.concatenate((np_coord, np_intensties), axis=1)
            #     comb_array = np.concatenate((comb_array, a), axis=1)

        c_s_new = self._compute_pca_components(labels=self._labels,subjects=self._tr_subjects,datasets=comb_shapes)

        intenses = self._compute_insities(labels=self._labels,subjects=self._tr_subjects)

        self._shape_coords = []
        self._intens_coords = []
        for i in range(len(c_s_new)):

            len_cur_shape = c_s_new[i].shape[1]
            len_intenses = intenses[i].shape[1]
            if i == 0:
                self._shape_coords.append([0,len_cur_shape])
                self._intens_coords.append([len_cur_shape,len_cur_shape+len_intenses])
            else:
                self._shape_coords.append([self._intens_coords[i-1][1]
                                              ,self._intens_coords[i-1][1] + len_cur_shape])
                self._intens_coords.append( [self._shape_coords[i][1],
                                            self._shape_coords[i][1] + len_intenses])

            if comb_array is None:
                comb_array = np.concatenate((c_s_new[i],intenses[i]),axis=1)
            else:
                a = np.concatenate((c_s_new[i],intenses[i]),axis=1)
                comb_array = np.concatenate((comb_array, a), axis=1)


        self.shape_data = c_s_new
        self.intens_data = intenses
        self._median_all = distros.NormalDistribution.calculate_median(c_s_new)
        jd = distros.NormalDistribution(comb_array)

        kde_combined = jd
        # save estimators
        self._kdes = kde_combined

    def _parse_label(self, subj, label):

        points = []
        with open(subj + os.sep + label + "_profiles.csv", 'r') as file:
            reader = csv.reader(file)

            for row in reader:
                point_coords = []
                for i in range(settings.settings.discretisation):
                    vox = []
                    vox.append(float(row[3 * i]))
                    vox.append(float(row[3 * i + 1]))
                    vox.append(float(row[3 * i + 2]))
                    point_coords.append(vox)
                rest = row[3 * settings.settings.discretisation:]

                for i in range(settings.settings.discretisation):
                    rest[i] = distutils.util.strtobool(rest[i]) > 0

                bl_mask = rest[:settings.settings.discretisation]
                rest = rest[settings.settings.discretisation:]
                for i in range(len(rest)):
                    rest[i] = float(rest[i])
                points.append([point_coords, bl_mask, rest])

        return points

        pass

    def _read_labels(self, labels, train_subjects):
        res = [[] for i in range(len(labels))]
        self._tr_subjects = train_subjects
        for i in range(len(train_subjects)):
            for j in range(len(labels)):
                res[j].append(self._parse_label(train_subjects[i], labels[j]))  # sort for labels.
                # 1 coords of norm 2 touched finish 3 end
                pass

        return res

    def __init__(self, train_subjects, labels, segmentation_conf):

        self._labels = labels

        self.use_constraint = segmentation_conf['use_constraint']

        self._original_data = self._read_labels(labels=labels, train_subjects=train_subjects)

        self._construct_pdm()

    def save_pdm(self, file_name, save_orig=False):
        try:
            os.remove(file_name)
        except:
            pass
        if (~ save_orig):
            self._original_data = None
        f = open(file_name, 'wb')
        pickle.dump(self, f)

    def get_kdes(self):
        return self._kdes.copy()

    @staticmethod
    def read_pdm(file_name):
        f = open(file_name, 'rb')
        res = pickle.load(f)
        return res

    def recompute_conditional_shape_int_distribution(self, num_of_pts):
        res = []
        self._label_kde = settings.settings.labels_to_segment
        for i in range(len(self._label_kde)):
            # recompute coord pos
            ind = self._labels.index(self._label_kde[i])

            # num_intensity_coords = self._intens_coords[ind][1]-self._intens_coords[ind][0]
            # num_per_structure = 3 * num_of_pts + num_intensity_coords
            #
            # mean_shape = self._kdes.distr.mean[self._shape_coords[ind][0]:self._shape_coords[ind][1]]
            # mean_intens = self._kdes.distr.mean[self._intens_coords[ind][0]: self._intens_coords[ind][1]]
            #
            #
            # mean_all1 = self._kdes.distr.mean[self._shape_coords[ind][0]:self._intens_coords[ind][1]]
            # cov_all1 = self._kdes.distr.cov[self._shape_coords[ind][0]:self._intens_coords[ind][1],
            #            self._shape_coords[ind][0]:self._intens_coords[ind][1]]
            norm_cond = distros.NormalConditional(data_main=self.shape_data[ind]
                                                  ,data_condition=self.intens_data[ind],
                                                  tol=10)
            norm_cond_b = distros.ProductJoined_ShInt_Distribution(data_main=self.shape_data[ind]
                                                  ,data_condition=self.intens_data[ind])

            res.append([norm_cond, norm_cond_b,self._median_all[ind]])

        return res

    def _recompute_cond_matrices(self, key, value, cov_all, mean_all, num_pts):
        ##consts initialisation

        #els_in_structure = num_pts * 3 + num_pts * settings.settings.discretisation
        #els_pts_per_struct = num_pts * 3
        #################
        keys = key.split(',')
        values = value.split(',')

        indices_keys = []
        indices_values = []

        for el in keys:
            ind = self._labels.index(el)
            indices_keys.append(ind)
        pass

        for el in values:
            ind = self._labels.index(el)
            indices_values.append(ind)

        ##join values of conditional structure shape representation
        s2_data = None
        for ind in indices_keys:
            if s2_data is None:
                s2_data = self.shape_data[ind]
            else:
                s2_data = np.concatenate((s2_data,self.shape_data[ind]),axis=-1)

        s1_data = None
        i1_data = None
        for ind in indices_values:
            if s1_data is None:
                s1_data = self.shape_data[ind]
            else:
                s1_data = np.concatenate((s1_data,self.shape_data[ind]),axis=-1)
            if i1_data is None:
                i1_data = self.intens_data[ind]
            else:
                i1_data = np.concatenate( (i1_data,self.intens_data[ind]),axis=-1)

        return [i1_data,s1_data,s2_data]
        #
        #
        #
        # range_shape_indices = []
        # for i in indices_keys:
        #     tmp = []
        #     tmp.append(self._shape_coords[i][0])
        #     tmp.append(self._shape_coords[i][1])
        #     range_shape_indices.append(tmp)
        #
        # #####join shapes of depended structures
        # range_dependent_shapes = []
        # range_dependent_intensities = []
        # for i in indices_values:
        #     # s
        #     tmp = []
        #     tmp.append(self._shape_coords[i][0])
        #     tmp.append(self._shape_coords[i][1])
        #     range_dependent_shapes.append(tmp)
        #     # int
        #     tmp1 = []
        #     tmp1.append(self._intens_coords[i][0])
        #     tmp1.append(self._intens_coords[i][1])
        #     range_dependent_intensities.append(tmp1)
        #
        # #####################################################
        # ######## join shapes into two representations
        # ######## result two covariance matrices
        #
        # shape_shape_mat_indices_1 = range_dependent_shapes + range_shape_indices
        #
        # shape_shape_res = []
        # for i in range(len(shape_shape_mat_indices_1)):
        #     tmp = []
        #     for j in range(len(shape_shape_mat_indices_1)):
        #         tmp.append([shape_shape_mat_indices_1[i], shape_shape_mat_indices_1[j]])
        #     shape_shape_res.append(tmp)
        #
        # shape_shape_cov_mat = None
        # for i in range(len(shape_shape_mat_indices_1)):
        #     tmp = None
        #     for j in range(len(shape_shape_mat_indices_1)):
        #         ss1_0 = shape_shape_res[i][j][0][0]
        #         ss1_1 = shape_shape_res[i][j][0][1]
        #         ss2_0 = shape_shape_res[i][j][1][0]
        #         ss2_1 = shape_shape_res[i][j][1][1]
        #         if tmp is None:
        #             tmp = cov_all[ss1_0:ss1_1, ss2_0:ss2_1]
        #         else:
        #             tmp = np.concatenate((tmp, cov_all[ss1_0:ss1_1, ss2_0:ss2_1]), axis=1)
        #     if shape_shape_cov_mat is None:
        #         shape_shape_cov_mat = tmp
        #     else:
        #         shape_shape_cov_mat = np.concatenate((shape_shape_cov_mat, tmp))
        #
        # mean_sc1 = None
        # mean_sc2 = None
        #
        # for i in range(len(range_dependent_shapes)):
        #     sc1 = range_dependent_shapes[i][0]
        #     sc2 = range_dependent_shapes[i][1]
        #     if mean_sc1 is None:
        #         mean_sc1 = mean_all[sc1:sc2]
        #     else:
        #         mean_sc1 = np.concatenate((mean_sc1, mean_all[sc1:sc2]))
        #
        # for i in range(len(range_shape_indices)):
        #     sc1 = range_shape_indices[i][0]
        #     sc2 = range_shape_indices[i][1]
        #     if mean_sc2 is None:
        #         mean_sc2 = mean_all[sc1:sc2]
        #     else:
        #         mean_sc2 = np.concatenate((mean_sc2, mean_all[sc1:sc2]))
        #
        # ############################################################################
        # ############################################################################
        # ##################### Shape INTENSITY CONDITIONAL COV MAT
        # ############################################################
        #
        # shape_int_indices = range_dependent_intensities + range_dependent_shapes
        #
        # shape_int_mat_indices_1 = []
        #
        # for i in range(len(shape_int_indices)):
        #     tmp = []
        #     for j in range(len(shape_int_indices)):
        #         tmp.append([shape_int_indices[i], shape_int_indices[j]])
        #     shape_int_mat_indices_1.append(tmp)
        #
        # shape_int_mat = None
        #
        # for i in range(len(shape_int_indices)):
        #     tmp = None
        #     for j in range(len(shape_int_indices)):
        #         ss1_0 = shape_int_mat_indices_1[i][j][0][0]
        #         ss1_1 = shape_int_mat_indices_1[i][j][0][1]
        #         ss2_0 = shape_int_mat_indices_1[i][j][1][0]
        #         ss2_1 = shape_int_mat_indices_1[i][j][1][1]
        #
        #         if tmp is None:
        #             tmp = cov_all[ss1_0:ss1_1, ss2_0:ss2_1]
        #         else:
        #             tmp = np.concatenate((tmp, cov_all[ss1_0:ss1_1, ss2_0:ss2_1]), axis=1)
        #     if shape_int_mat is None:
        #         shape_int_mat = tmp
        #     else:
        #         shape_int_mat = np.concatenate((shape_int_mat, tmp))
        # # means
        # mean_si1 = None
        # mean_si2 = None
        #
        # for i in range(len(range_dependent_shapes)):
        #     sc1 = range_dependent_shapes[i][0]
        #     sc2 = range_dependent_shapes[i][1]
        #     if mean_si2 is None:
        #         mean_si2 = mean_all[sc1:sc2]
        #     else:
        #         mean_si2 = np.concatenate((mean_si2, mean_all[sc1:sc2]))
        #
        # for i in range(len(range_dependent_intensities)):
        #     sc1 = range_dependent_intensities[i][0]
        #     sc2 = range_dependent_intensities[i][1]
        #     if mean_si1 is None:
        #         mean_si1 = mean_all[sc1:sc2]
        #     else:
        #         mean_si1 = np.concatenate((mean_si1, mean_all[sc1:sc2]))
        #
        # return [[mean_sc1, mean_sc2, shape_shape_cov_mat], [mean_si1, mean_si2, shape_int_mat]]

    def recompute_conditional_structure_structure(self, num_of_pts):
        res = []

        if not settings.settings.dependent_constraint:
            return
        for i in range(len(settings.settings.dependent_constraint.keys())):
            key = list(settings.settings.dependent_constraint.keys())[i]
            [ i1_data, s1_data, s2_data] = self._recompute_cond_matrices(key, settings.settings.dependent_constraint[key],
                                                                     self._kdes.distr.cov, self._kdes.distr.mean,
                                                                     num_of_pts)

            t_Res = distros.JointDependentDistribution(data_s1=s1_data,data_s2=s2_data,data_I1=i1_data)

            res.append(t_Res)
        return res


    def get_shape_pca(self, ind):
        return self._pca_shapes[ind]

    def get_intens_pca(self,ind):
        return  self._pca_intensities[ind]

    def get_shape_coords(self, ind):
        return self._shape_coords[ind]

    def get_intens_coords(self,ind):
        return  self._intens_coords[ind]