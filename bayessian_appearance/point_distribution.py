
import bayessian_appearance.settings as settings
import bayessian_appearance.distributions as distros
import os
import csv
import sklearn.neighbors as neighb
import distutils.util
import numpy as np
import pickle

import scipy.stats as scp_stats

class PointDistribution:
    use_constraint = None
    _original_data = None

    _labels = None


    _kdes = None
    #description of each label
    _label_kde = None

    #extrapolate to false
    def _prolongue_label(self, label_vec, mask_vec):
        res = label_vec.copy()
        i_f = mask_vec.index(True)
        # if min(mask_vec) == False:
        #     print("HAS FALSE")
        ##f
        for i in (range(i_f)):
            res[i]=res[i_f]
        return res




    def _construct_pdm(self):

        #find what we segmenting
        list_pos= [] #positions of labels which we segmenting
        for i in range(len(settings.settings.labels_to_segment)):
            pos = self._labels.index(settings.settings.labels_to_segment[i])
            list_pos.append(pos)

        pass
        self._label_kde = settings.settings.labels_to_segment
        posit_of_cent= int(settings.settings.discretisation/2)

        kde_combined = []
        # construct combined ONE PDM for ALL labels here
        comb_array = None
        for i in range(len(self._labels)):
            pdm_label_data = self._original_data[i]
            #get locations]


            # calculate cumulative coordinate of each coord. row sample col
            coordinates = []
            for j in range(len(pdm_label_data[0])):
                vec_j = [ x[j][0][posit_of_cent] for x in pdm_label_data ]
                coordinates.append(vec_j)


            #generate kdes for coords
            posit_kdes = []


            #here we form joint coordinates rows = sample col = coords (x y z)
            np_coord = None
            for j in range(len(coordinates)):
                if j == 0:
                    np_coord = np.array(coordinates[j])
                    continue
                np_coord= np.concatenate((np_coord,np.array(coordinates[j])),axis=1)


            #############INTENSITY KDES
            ##RECord KDE like as whole profile (ESTIMATE THROUGH Whoole profile)!!!
            kdes_norms = []
            #TODO add multiple modalities here
            num_of_mods = int(len(pdm_label_data[0][0][2]) / settings.settings.discretisation)


            intensity_profs = []
            intensity_kdes = []
            for vert in range(len(pdm_label_data[0])):
                profile = []
                for sub in range(len(pdm_label_data)):
                    single_profile = self._prolongue_label( label_vec=pdm_label_data[sub][vert][2],
                                                     mask_vec=pdm_label_data[sub][vert][1])
                    profile.append(single_profile)


                intensity_profs.append(profile)
                #i_kde = scp_stats.gaussian_kde(profile.transpose()) #TODO MAYBE ADD CONSTRAINTS FOR OUTSIDE


            #here we form joint intensity rows = sample col = coords (x y z)
            np_intensties  = None
            for j in range(len(pdm_label_data[0])):
                if j == 0:
                    np_intensties = np.array(intensity_profs[j])
                    continue
                np_intensties= np.concatenate((np_intensties,np.array(intensity_profs[j])),axis=1)


            # formulate shape+intensity distribution
            if comb_array is None:

                comb_array = np.concatenate((np_coord,np_intensties),axis=1)
            else:
                a = np.concatenate((np_coord,np_intensties),axis=1)
                comb_array = np.concatenate((comb_array,a),axis=1)

        jd = distros.NormalDistribution(comb_array)

        kde_combined = jd
        #save estimators
        self._kdes = kde_combined






    def _parse_label(self,subj,label):

        points = []
        with open(subj + os.sep + label + "_profiles.csv",'r') as file:
            reader = csv.reader(file)

            for row in reader:
                point_coords = []
                for i in range(settings.settings.discretisation):
                    vox = []
                    vox.append(float(row[3*i]))
                    vox.append(float(row[3*i + 1]))
                    vox.append(float(row[3 * i + 2]))
                    point_coords.append(vox)
                rest = row[3*settings.settings.discretisation:]

                for i in range(settings.settings.discretisation):
                    rest[i] = distutils.util.strtobool(rest[i]) > 0

                bl_mask = rest[:settings.settings.discretisation]
                rest = rest[settings.settings.discretisation:]
                for i in range(len(rest)):
                    rest[i] = float(rest[i])
                points.append( [ point_coords, bl_mask,rest])

        return points




        pass
    def _read_labels(self, labels,train_subjects):
        res= [[] for i in range(len(labels))]

        for i in range(len(train_subjects)):
            for j in range(len(labels)):

                res[j].append(self._parse_label(train_subjects[i],labels[j])) #sort for labels.
                #1 coords of norm 2 touched finish 3 end
                pass






        return res
    def __init__(self, train_subjects, labels, segmentation_conf):

        self._labels = labels

        self.use_constraint = segmentation_conf['use_constraint']

        self._original_data = self._read_labels(labels=labels,train_subjects=train_subjects)

        self._construct_pdm()


    def save_pdm(self, file_name, save_orig = False):
        try:
            os.remove(file_name)
        except:
            pass
        if (~ save_orig):
            self._original_data = None
        f = open(file_name,'wb')
        pickle.dump(self,f)


    def get_kdes(self):
        return self._kdes.copy()

    @staticmethod
    def read_pdm( file_name):
        f = open(file_name,'rb')
        res = pickle.load(f)
        return res


    def recompute_conditional_shape_int_distribution(self, num_of_pts):
        res = []
        for i in range(len(self._label_kde)):

            #recompute coord pos
            ind = self._labels.index(self._label_kde[i])

            num_intensity_coords = settings.settings.discretisation * num_of_pts
            num_per_structure = 3* num_of_pts + num_intensity_coords


            mn = self._kdes.distr.mean[i*num_per_structure:i*num_per_structure+ 3*num_of_pts]
            mn2 = self._kdes.distr.mean[num_per_structure*i+ 3*num_of_pts: num_per_structure *(i+1)]
            cv = self._kdes.distr.cov[i*num_per_structure : i*num_per_structure+ 3*num_of_pts
                                        ,i*num_per_structure:i*num_per_structure + 3*num_of_pts]

            mean_all1  = self._kdes.distr.mean[num_per_structure*i:num_per_structure*(i+1)]
            cov_all1 = self._kdes.distr.cov[num_per_structure*i:num_per_structure*(i+1),num_per_structure*i:num_per_structure*(i+1)]
            norm_cond = distros.NormalConditional(mean1=mn,mean2=mn2,cov_all=cov_all1,num_of_pts=num_of_pts*3)
            norm_cond_b = distros.NormalConditionalBayes(mean_all=mean_all1,cov_all=cov_all1,num_of_pts=3*num_of_pts)

            res.append([norm_cond,norm_cond_b])

        return res


    def _recompute_cond_matrices(self,key,value,cov_all, mean_all,num_pts):
        ##consts initialisation

        els_in_structure = num_pts*3 + num_pts*settings.settings.discretisation
        els_pts_per_struct = num_pts*3
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

        range_shape_indices = []
        for i in indices_keys:
            tmp = []
            tmp.append( i* els_in_structure )
            tmp.append(i*els_in_structure + els_pts_per_struct)
            range_shape_indices.append(tmp)



        #####join shapes of depended structures
        range_dependent_shapes=[]
        range_dependent_intensities = []
        for i in indices_values:
            #s
            tmp = []
            tmp.append( i* els_in_structure )
            tmp.append(i*els_in_structure + els_pts_per_struct)
            range_dependent_shapes.append(tmp)
            #int
            tmp = []
            tmp.append( i*els_in_structure + els_pts_per_struct )
            tmp.append((i+1)*els_in_structure)
            range_dependent_intensities.append(tmp)

        #####################################################
        ######## join shapes into two representations
        ######## result two covariance matrices

        shape_shape_mat_indices_1 = range_dependent_shapes + range_shape_indices

        shape_shape_res = []
        for i in range( len(shape_shape_mat_indices_1)):
            tmp = []
            for j in range(len(shape_shape_mat_indices_1)):
                tmp.append([shape_shape_mat_indices_1[i],shape_shape_mat_indices_1[j]])
            shape_shape_res.append(tmp)

        shape_shape_cov_mat = None
        for i in range( len(shape_shape_mat_indices_1)):
            tmp = None
            for j in range(len(shape_shape_mat_indices_1)):
                ss1_0 = shape_shape_res[i][j][0][0]
                ss1_1 = shape_shape_res[i][j][0][1]
                ss2_0 = shape_shape_res[i][j][1][0]
                ss2_1 = shape_shape_res[i][j][1][1]
                if tmp is None:
                    tmp = cov_all[ss1_0:ss1_1,ss2_0:ss2_1]
                else:
                    tmp = np.concatenate((tmp, cov_all[ss1_0:ss1_1,ss2_0:ss2_1]),axis=1)
            if shape_shape_cov_mat is None:
                shape_shape_cov_mat = tmp
            else:
                shape_shape_cov_mat = np.concatenate((shape_shape_cov_mat,tmp))

        mean_sc1 = None
        mean_sc2 = None

        for i in range(len(range_dependent_shapes)):
            sc1 = range_dependent_shapes[i][0]
            sc2 = range_dependent_shapes[i][1]
            if mean_sc1 is None:
                mean_sc1 = mean_all[sc1:sc2]
            else:
                mean_sc1 = np.concatenate((mean_sc1,mean_all[sc1:sc2]) )

        for i in range(len(range_shape_indices)):
            sc1 = range_shape_indices[i][0]
            sc2 = range_shape_indices[i][1]
            if mean_sc2 is None:
                mean_sc2 = mean_all[sc1:sc2]
            else:
                mean_sc2 = np.concatenate((mean_sc2,mean_all[sc1:sc2]) )

        ############################################################################
        ############################################################################
        ##################### Shape INTENSITY CONDITIONAL COV MAT
        ############################################################

        shape_int_indices=  range_dependent_intensities + range_dependent_shapes



        shape_int_mat_indices_1 = []

        for i in range( len(shape_int_indices)):
            tmp = []
            for j in range(len(shape_int_indices)):
                tmp.append([shape_int_indices[i],shape_int_indices[j]])
            shape_int_mat_indices_1.append(tmp)

        shape_int_mat = None

        for i in range( len(shape_int_indices)):
            tmp = None
            for j in range(len(shape_int_indices)):
                ss1_0 = shape_int_mat_indices_1[i][j][0][0]
                ss1_1 = shape_int_mat_indices_1[i][j][0][1]
                ss2_0 = shape_int_mat_indices_1[i][j][1][0]
                ss2_1 = shape_int_mat_indices_1[i][j][1][1]

                if tmp is None:
                    tmp = cov_all[ss1_0:ss1_1, ss2_0:ss2_1]
                else:
                    tmp = np.concatenate((tmp, cov_all[ss1_0:ss1_1, ss2_0:ss2_1]), axis=1)
            if shape_int_mat is None:
                shape_int_mat = tmp
            else:
                shape_int_mat = np.concatenate((shape_int_mat, tmp))
        #means
        mean_si1 = None
        mean_si2 = None

        for i in range(len(range_dependent_shapes)):
            sc1 = range_dependent_shapes[i][0]
            sc2 = range_dependent_shapes[i][1]
            if mean_si2 is None:
                mean_si2 = mean_all[sc1:sc2]
            else:
                mean_si2 = np.concatenate((mean_si2, mean_all[sc1:sc2]))

        for i in range(len(range_dependent_intensities)):
            sc1 = range_dependent_intensities[i][0]
            sc2 = range_dependent_intensities[i][1]
            if mean_si1 is None:
                mean_si1 = mean_all[sc1:sc2]
            else:
                mean_si1 = np.concatenate((mean_si1, mean_all[sc1:sc2]))


        return [ [mean_sc1,mean_sc2,shape_shape_cov_mat] , [mean_si1,mean_si2,shape_int_mat]   ]





    def recompute_conditional_structure_structure(self,num_of_pts):
        res = []

        for i in range(len(settings.settings.dependent_constraint.keys())):
            key = list(settings.settings.dependent_constraint.keys())[i]
            [shape_shape, shape_int] = self._recompute_cond_matrices(key,settings.settings.dependent_constraint[key],
                                                                      self._kdes.distr.cov,self._kdes.distr.mean,num_of_pts)


            t_Res = distros.JointDependentDistribution(mean_s_1=shape_shape[0],mean_s_2=shape_shape[1],cov_s=shape_shape[2],
                                                       mean_si1=shape_int[0],mean_si2=shape_int[1],cov_si=shape_int[2])


            res.append(t_Res)
        return res




















