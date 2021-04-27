resim=$2
struct_image=$1
N4BiasFieldCorrection --image-dimensionality 3  --input-image $struct_image --output $resim --shrink-factor 4 -b --convergence [300x300x300x300,0.000001]