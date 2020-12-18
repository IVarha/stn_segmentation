
import nibabel as nib


import sys

import numpy as np
import os


def get_coord_voxel(coord_orig,orig_transf):
    # get v
    vox = np.array(coord_orig + [1])

    res = np.dot(orig_transf,vox)
    return res[:3]



def imageHistogram( vol,  bins, mn,mx, mask):

    if mx<=mn: return -1

    valsize =0
    hist = 0
    fA = (bins)/(mx-mn)
    fB =  (bins * (-mn)) / (mx-mn)
    v2 = vol[mask==1]

    hist = [0]
    for el in v2:
        val = min( fA *el + fB,bins-1 ) + 1
        val2 = max(0,val )
        hist.append(val2)
        valsize +=1
    return [valsize, hist]

    # for i in range(vol.shape[0]):
    #     for j in range(vol.shape[1]):
    #         for k in range( vol.shape[2]):
    #             val = min(fa*(vol[ ]) )



def calc_robust_minmax(image,mask):

    res = []


    HISTOGRAM_BINS = 1000

    hist = np.zeros((HISTOGRAM_BINS,1))
    MAX_PASSES=10
    top_bin=0
    bottom_bin=0
    count=1
    pass1=1

    lowest_bin=0
    highest_bin=HISTOGRAM_BINS-1
    thresh98 = 0
    thresh2 = 0
    mn = image[mask==1].min()
    mx = image[mask == 1].max()

    while ((pass1 == 1) or ((thresh98 - thresh2) < (( (mx - mn)) / 10.0)) ) :

        if pass1>1:
            bottom_bin = max(bottom_bin - 1, 0)
            top_bin = max(top_bin + 1, HISTOGRAM_BINS - 1)

            tmpmin = (mn + (bottom_bin /HISTOGRAM_BINS) * (mx - mn) )
            mx = (mn + ((top_bin + 1) / HISTOGRAM_BINS) * (mx - mn))
            mn = tmpmin

        if (pass1 == MAX_PASSES or mn == mx):
            mn = image[mask == 1].min()
            mx = image[mask == 1].max()
            pass

        [validsize,hist] = imageHistogram(vol=image,bins=HISTOGRAM_BINS,mn=mn,mx=mx,mask=mask)

        if validsize <1:
            return [mn,mx]

        if (pass1 == MAX_PASSES):
            validsize -= round(hist[lowest_bin + 1]) + round(hist[highest_bin + 1])
            lowest_bin +=1
            highest_bin -=1

        if (validsize<0):
            thresh2=thresh98=mn
            break

        fA = (mx - mn) / (HISTOGRAM_BINS)

        bottom_bin = lowest_bin
        count =0
        while(count < validsize/50):
            count+= round(hist[bottom_bin+1])
            bottom_bin +=1
        bottom_bin-=1
        thresh2 = mn + (bottom_bin * fA)

        count = 0
        top_bin = highest_bin
        while(count < validsize/50):
            count+= round(hist[top_bin+1])
            top_bin -=1
        top_bin+=1
        thresh98 = mn + (top_bin + 1) * fA
        if (pass1 == MAX_PASSES): break
        pass1+=1

    return [thresh2,thresh98]




def calc_robust_images(image, mask):

    #ths = calc_robust_minmax(image,mask)
    a =image[mask==1].mean()
    im = image / a
    #im[mask==1] = (im[mask==1] - ths[0])/(ths[1]-ths[0])
    #im[mask!=1] = 0
    return im




if __name__ == '__main__':
    t2_in = sys.argv[1]
    mask = sys.argv[2]
    outp = sys.argv[3]


    #read pve segmentation
    mask_file = nib.load(mask)

    t2_file = nib.load(t2_in)

    pve_transf = mask_file.affine
    t2_transf = t2_file.affine

    mask_im = mask_file.get_fdata()

    mask_t = np.zeros(t2_file.get_fdata().shape)
    im_w_2_v =np.linalg.inv(t2_transf)
    for i in range(mask_im.shape[0]):
        for j in range(mask_im.shape[1]):
            for k in range(mask_im.shape[2]):
                if mask_im[i,j,k] == 1:
                    world = get_coord_voxel([i,j,k],pve_transf)
                    coords_mask = get_coord_voxel(list(world), im_w_2_v)
                    pos = np.round(coords_mask).astype(np.int)
                    try:
                        mask_t[pos[0], pos[1], pos[2]] = 1
                    except:
                        pass







    res = calc_robust_images(image=t2_file.get_fdata(), mask=mask_t)


    # res = create_mask(t2_img=t2_file.get_fdata(),t2_transf=t2_transf,
    #                   pve_img=pve_file.get_fdata(),pve_transf=pve_transf)

    nif2 = nib.Nifti1Image(res, t2_transf)
    try:
        os.remove(outp)
    except:
        pass
    nib.save(nif2,outp)

    # Initialize the layout
    print(1222)
