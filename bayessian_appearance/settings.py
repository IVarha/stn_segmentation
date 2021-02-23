

class Settings:
    use_constraint = None
    labels_to_segment = None
    dependent_constraint = None
    joint_labels=None
    norm_length = None
    discretisation = None
    all_labels = None
    atlas_dir = None
    modalities = None


    pca_precision = 0.995
    random_state = 1

    def __init__(self):
        pass



settings = Settings()




def setup_settings(segmentation_settings):

    USE_CONSTRAINT = segmentation_settings['use_constraint']
    pass