class BaseOptions(object):
    def __init__(self, para_dict=None) -> None:
        """
        Initialize the BaseOptions object.

        Args:
            para_dict (dict): A dictionary containing training parameters. If None, default values will be used.
        """
        super().__init__()

        self.bg_type = "black"

        self.pose_dims = 12
        self.shape_dims = 100
        self.expression_dims = 100
        self.albedo_dims = 100
        self.illumination_dims = 27

        self.num_sample_coarse = 32
        self.num_sample_fine = 128

        self.world_z1 = 0.2
        self.world_z2 = -0.2
        self.mlp_hidden_nchannels = 384

        if para_dict is None:
            self.featmap_size = 16
            self.featmap_nc = 128  
            self.pred_img_size = 128
        else:
            self.featmap_size = para_dict["featmap_size"]
            self.featmap_nc = para_dict["featmap_nc"]
            self.pred_img_size = para_dict["pred_img_size"]
