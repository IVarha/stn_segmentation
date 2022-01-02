# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import sys

import bayessian_appearance.utils as util
import vtk

import bayessian_appearance.point_distribution as pd
import bayessian_appearance.settings as settings
import os
import numpy as np
import datetime
import sklearn.cluster as clst

from mpl_toolkits.mplot3d import Axes3D,art3d

import matplotlib.pyplot as plt
import sklearn.covariance as cov


def process_STN(data,workdir,stn=[2,3],num_clusters =3):

    first_cluster = None


    clusters_res = []
    for st_i in stn:
        proc_int = data[st_i]

        all_subjs_classes = []

        all_subjs_centers = []
        ###########calc subjects
        for i in  range(len(proc_int)):

            #calc 1 subject

            subj_data = []
            for j in range(len(proc_int[i])):
                subj_data.append(proc_int[i][j][2])

            subj_data = np.array(subj_data)

            subj_clst = clst.KMeans(n_clusters=num_clusters)
            subj_clst.fit(X=subj_data)
            all_subjs_classes.append(subj_clst.labels_)
            all_subjs_centers.append(subj_clst.cluster_centers_)
        #recalculate clusters
        cent = all_subjs_centers[0]
        for cnt_i in range(1,len(all_subjs_centers)):
            temp_cluster = [0] * len(all_subjs_classes[0])
            for i_j in range(num_clusters):
                cent_val1 = np.array(cent[i_j])
                dists = []
                for j_j in range(num_clusters):
                    cent_val2 = np.array(all_subjs_centers[cnt_i][j_j])
                    dists.append(np.linalg.norm(cent_val2-cent_val1))
                ind_closest_clust = dists.index(min(dists))
                for i_vox in range(len(temp_cluster)):
                    if all_subjs_classes[cnt_i][i_vox] == ind_closest_clust:
                        temp_cluster[i_vox] = i_j
            all_subjs_classes[cnt_i]= temp_cluster

        ############
        ##########calculate max overlap clusters

        best_clusters = []

        for i_vox in range(len(all_subjs_classes[0])):


            ar_classes = [0]* num_clusters
            for i_sub in range(len(all_subjs_classes)):
                ar_classes[all_subjs_classes[i_sub][i_vox]]+=1

            best_clusters.append(ar_classes.index(max(ar_classes)))



        res_ints = [None] *len(proc_int[0])
        # calculate vertices
        for i in  range(len(proc_int)):


            for j in range(len(proc_int[i])):
                if res_ints[j]==None:
                    res_ints[j] = []
                res_ints[j].append(proc_int[i][j][2])

        res_joint_array = None
        for i in range(len(res_ints)):
            if res_joint_array is None:
                res_joint_array = np.array(res_ints[i])
            else:
                res_joint_array = np.concatenate((res_joint_array,np.array(res_ints[i])),axis=-1)

        r_stat = cov.EllipticEnvelope(support_fraction=0.3).fit(res_joint_array)
        mn_vec = r_stat.location_
        mn_vec = mn_vec.reshape((len(proc_int[0]),int(mn_vec.shape[0]/len(proc_int[0]))))
        # calculate mean


        # mn_vec = [None] *len(proc_int[0])
        # for i in  range(len(res_ints)):
        #     res_ints[i] = np.array(res_ints[i])
        #     mn_vec[i] = np.array(res_ints[i].mean(axis=0))

        mn_vec = np.array(mn_vec)

        #ms = clst.MeanShift().fit(X=mn_vec)
        ms = None


        if first_cluster is None:
            ms = clst.KMeans(n_clusters=num_clusters).fit(X=mn_vec)
            first_cluster = ms
        else:
            ms = clst.KMeans(n_clusters=num_clusters,init=first_cluster.cluster_centers_).fit(X=mn_vec)
            clusters = set(ms.labels_)
            # cl_centrs = []
            # temp_cluster = [0] * len(all_subjs_classes[0])
            # sub_class = ms.labels_
            # for i_j in range(ms.cluster_centers_.shape[0]):
            #     cent_val1 = first_cluster.cluster_centers_[i_j,:]
            #     dists = []
            #     for j_j in range(ms.cluster_centers_.shape[0]):
            #         cent_val2 = ms.cluster_centers_[j_j,:]
            #         dists.append(np.linalg.norm(cent_val2 - cent_val1))
            #     ind_closest_clust = dists.index(min(dists))
            #     cl_centrs.append(ms.cluster_centers_[ind_closest_clust,:].tolist())
            #     for i_vox in range(len(sub_class)):
            #         if sub_class[i_vox] == ind_closest_clust:
            #             temp_cluster[i_vox] = i_j
            # ms.labels_ = temp_cluster
            # ms.cluster_centers_ = np.array(cl_centrs)



        clusters = set(ms.labels_)
        #
        dt = 6/14
        x_S = [-3+i*dt for i in range(15)]
        #x_S =
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for cl in clusters:
            ax.plot(x_S,ms.cluster_centers_[cl],label=str(cl),color=settings.settings.colors[cl])
            # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.legend()
        plt.savefig(workdir +os.sep+ str(st_i)+"_clusters.svg")

        clusters_res.append([i for i in range(len(clusters))])
        with open(workdir+os.sep+ str(st_i) + "clusters.txt","wt") as f:

            labls =ms.labels_
            res_str = ""
            for i in range(len(labls)-1):
                res_str += str(labls[i]) + ","
            res_str += str(labls[len(labls)-1])
            f.write(res_str)

    return clusters_res








def plot_clusters(meshes, clusters):

    for i in range(len(meshes)):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        fc = np.array(meshes[i].get_faces())
        v = meshes[i].get_unpacked_coords()
        v = np.array(v)
        v = v.reshape((int(v.shape[0] / 3), 3))
        # cluster to set
        st = set(clusters[i])

        pc = art3d.Poly3DCollection(v[fc], edgecolor="black")
        ax.add_collection3d(pc)
        cluster = np.array(clusters[i])
        for clust in range(len(st)):
            pts = v[cluster==clust,:]


            x = pts[:,0]
            y = pts[:, 1]
            z = pts[:, 2]







def main_proc(train, label_names, config_name, workdir):
    tr_subjects = util.read_subjects(train)
    labels = util.read_label_desc(label_names)

    mod = util.read_modalities_config(modalities_name)

    seg_cnf = util.read_segmentation_config(modalities_name)
    cnf = util.read_config_ini(config_name)
    print("processing started at")
    print(datetime.datetime.now())
    meshes = []
    pdm = pd.PointDistribution(train_subjects=tr_subjects, labels=labels, segmentation_conf=seg_cnf,construct=False)

    meshes = pdm.get_mean_meshes()
    print("processing finished at")
    od = pdm.get_original_data()
    print(datetime.datetime.now())

    meshes[2].save_obj(workdir + os.sep + "3_mean.obj")
    meshes[3].save_obj(workdir + os.sep + "4_mean.obj")
    clsters = process_STN(od,workdir,num_clusters=2)


    plot_clusters(meshes=[meshes[2],meshes[3]],clusters=clsters)




    print(1)



    # read im

    # meshes = np.array(meshes)
    m_mx = []


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_subjects_file = sys.argv[1]
    labels_desc_file = sys.argv[2]
    conf_file = sys.argv[3]
    outp = sys.argv[4]
    modalities_name = sys.argv[5]
    a = util.read_segmentation_config(modalities_name)
    main_proc(train_subjects_file, labels_desc_file, conf_file, workdir=outp)

    # Print some basic information about the layout

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
