
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

        ##f
        for i in (range(i_f)):
            res[i]=res[i_f]
        return res




    def _construct_pdm(self):

        #find what we segmenting
        list_pos= []
        for i in range(len(settings.settings.labels_to_segment)):
            pos = self._labels.index(settings.settings.labels_to_segment[i])
            list_pos.append(pos)

        pass
        self._label_kde = settings.settings.labels_to_segment
        posit_of_cent= int(settings.settings.discretisation/2)

        kde_combined = []
        # construct PDM for each segmented label!!!!! HERE
        for i in range(len(list_pos)):
            pdm_label_data = self._original_data[list_pos[i]]
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
            jd = distros.NormalDistribution(np.concatenate((np_coord,np_intensties),axis=1))

            kde_combined.append(jd)
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


    def save_pdm(self, file_name):
        try:
            os.remove(file_name)
        except:
            pass
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
        for i in range(len(self._kdes)):
            mn = self._kdes[i].distr.mean[:3*num_of_pts]
            mn2 = self._kdes[i].distr.mean[3*num_of_pts:]
            cv = self._kdes[i].distr.cov[:3*num_of_pts,:3*num_of_pts]

            cv2 = np.dot(self._kdes[i].distr.cov_info.U,self._kdes[i].distr.cov_info.U.transpose())[:3 * num_of_pts, 3 * num_of_pts:]


            norm_cond = distros.NormalConditional(mean1=mn,mean2=mn2,cov11=cv,prec12=cv2)
            norm_cond_b = distros.NormalConditionalBayes(mean_all=self._kdes[i].distr.mean,cov_all=self._kdes[i].distr.cov,num_of_pts=3*num_of_pts)

            res.append([norm_cond,norm_cond_b])

        return res















