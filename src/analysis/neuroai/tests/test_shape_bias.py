from neuroai.evaluation.shape_bias_eval import get_shape_bias

if __name__ == '__main__':

    model_name = 'resnet50_DVD_B' #* Use naming in the form architecture + suffix (e.g., resnet50+XXX) so the architecture is automatically detected.
    evaluation = True # If run evaluation for the first time, set to True, else set to False to load the previous results
    load_checkpoint_path = '/share/klab/lzejin/lzejin/codebase/P001_dvd_gpus/logs/resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1/weights/checkpoint_best.pth' #None
    dataset_trained_on = ['ecoset', 'imagenet'][0]
    data_normalization_type = ["0-1", "mean-std"][0] #* #* Preprocessing normalization in 0-1 range, default is mean-std
    matrixplot_savedir = None

    avg_shape_bias, shape_bias_across, classes, shape_accs, texture_accs = get_shape_bias(model_name, evaluation = evaluation, load_checkpoint_path= load_checkpoint_path, dataset_trained_on = dataset_trained_on, data_normalization_type = data_normalization_type, matrixplot_savedir=matrixplot_savedir)
    print(f"shape bias : {avg_shape_bias}")
