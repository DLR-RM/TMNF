_base_ = ['../../feat_ext/faster_rcnn_r50_fpn_cocoOS_clsLogits.py']
flow_params=dict(
    random_seed = 3,
	num_cls = 51,
	flow_type = "rnvp", # "rnvp", "residual", "nsf_ar"
	input_dim = 51,
	blocks = 2,
	hidden_dim = 128,
	num_layers_st_net = 4,
	permutation = "affine", # "affine", "permute", "lu_permute"
	coupling = "affine", # String, "affine", "additive" 
	scale_map = "exp", # "sigmoid", "exp", "sigmoid_inv"
	lipschitz_const = 1.0, # for residual flow
	init_zeros = True, # Bool, flag whether to initialize last layer of NNs with zeros
	num_bins = 100, # for nsf_ar
	actnorm = True,
	dropout = False,
	prior_type = "gmm_cls_ib", # "gauss", "gmm", "resampled"
	######## base distribution specific ########
	base_learn_mean_var = False,
	############# for IB loss #############
	ib_beta=50.,
    ib_sigma=0.5,
	############# gmm base distribution specific #############
	base_n_modes = 51,  # number of cls
	base_loc_scale = 10, 
	######## training specific ########
	num_class = 51,
	batch_size = 1024,
	max_epochs = 200,
	val_freq = 5,
	early_stop_tolerance=0, # 0 to train until max_epochs, otherwise (number of iter+1) to stop when the model doesn't improve
	lr = 1e-4, # 5e-4, 1e-5
	weight_decay = 1e-5, # 5e-5, 1e-5, 1e-8
	max_norm = None, # 1e1, 5e1, 1e2, None
	optm = "Adamax", # SGD, Adam, "Adamax"
)