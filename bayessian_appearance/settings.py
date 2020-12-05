







class Settings:
    use_constraint = None
    labels_to_segment = None
    dependent_constraint = None
    norm_length = None
    discretisation = None

    def __init__(self):
        pass



settings = Settings()




def setup_settings(segmentation_settings):

    USE_CONSTRAINT = segmentation_settings['use_constraint']
    pass