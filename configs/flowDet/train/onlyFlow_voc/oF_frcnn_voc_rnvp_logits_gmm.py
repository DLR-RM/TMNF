_base_ = ['../../feat_ext/faster_rcnn_r50_fpn_voc0712OS_clsLogits.py']
flow_params=dict(
    random_seed = 3,
	flow_type = "rnvp", # "rnvp", "residual", "nsf_ar"
	input_dim = 16,
	blocks = 16,
	hidden_dim = 64,
	num_layers_st_net = 4,
	permutation = "affine", # "affine", "permute", "lu_permute"
	coupling = "affine", # String, "affine", "additive" 
	scale_map = "exp", # "sigmoid", "exp", "sigmoid_inv"
	init_zeros = True, # Bool, flag whether to initialize last layer of NNs with zeros
	actnorm = True,
	dropout = False,
	prior_type = "gmm", # "gauss", "gmm", "resampled"
	######## base distribution specific ########
	base_learn_mean_var = False,
	############# gmm base distribution specific #############
	base_n_modes = 16, 
	base_loc_scale = 10., 
	######## training specific ########
	num_class = 16,
	batch_size = 1024,
	max_epochs = 400,
	val_freq = 5,
	early_stop_tolerance=0, # 0 to train until max_epochs, otherwise (number of iter+1) to stop when the model doesn't improve
	lr = 1e-4, # 5e-4
	weight_decay = 1e-5, # 5e-5, 1e-5, 1e-8
	max_norm = None, # 1e1, 5e1, 1e2, None
	optm = "Adamax", # SGD, Adam, "Adamax"
)