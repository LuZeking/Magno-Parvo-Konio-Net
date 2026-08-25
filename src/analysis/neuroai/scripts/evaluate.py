import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from neuroai import Plot, Evaluate
from neuroai import constants as c
from plotting_definition import plotting_definition_template
from itertools import product

def run_evaluation():

    # EVD models
    models = [  
                
                    #* Default models
                    # # 'alexnet', 
                    # 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
                    # 'squeezenet1_0', 'squeezenet1_1', 'densenet121', 'densenet169', 'densenet201',
                    # 'inception_v3', 'resnet18', 'resnet34', 
                    # # 'resnet50', 
                    # 'resnet101', 'resnet152',
                    # 'shufflenet_v2_x0_5', 'mobilenet_v2', 
                    # 'resnext50_32x4d', 'resnext101_32x8d',
                    # 'wide_resnet50_2', 'wide_resnet101_2', 'mnasnet0_5', 'mnasnet1_0',
                    # "simclr_resnet50x1"
                    # 'vit_small_patch16_224', 'vit_base_patch16_224', 'vit_large_patch16_224',
                    # 'ResNeXt101_32x16d_swsl', 'resnet50_swsl',
                    # 'BiTM_resnetv2_152x4', 'BiTM_resnetv2_152x2', 'BiTM_resnetv2_101x3',
                    # 'BiTM_resnetv2_101x1', 'BiTM_resnetv2_50x3','BiTM_resnetv2_50x1',
                    # 'transformer_L16_IN21K', 'transformer_B16_IN21K', 
                    # 'efficientnet_l2_noisy_student_475', 
                    # 'clipRN50_CCM_12m_all_epochs',
                    # 'clipRN50_CCM_12m_DVD',
                    # 'clipRN50',
                    # 'clip',

                    #* extra
                    # 'MoCo',
                    # 'PIRL',
                    # 'MoCoV2',
                    # 'InfoMin',
                    # 'resnet50_l2_eps0',
                    # 'resnet50_l2_eps0_5',
                    # 'resnet50_l2_eps1',
                    # 'resnet50_l2_eps3',
                    # 'resnet50_l2_eps5',
                    # 'InsDis'


                    # 'clipRN50', 



                    

                    # #* VLMs
                    # 'qwen2_vl_2B_instruct',
                    # 'qwen2_vl_7B_instruct',

        
                    # "llava1_5_7b",
                    #   "llava_1_5_13b_hf", 
                      ##"llava_1_6_mistral_7b", "llava_1_6_vicuna_13b", # "llava_1_6_34b"

                    # 'blip2_opt_2_7b','blip2_flan_t5_xxl',

                    # 'qwen2_5_vl_7B_instruct',
                      # '',
                    # '',
                    # 'llava_1_5_7b_hf',
                    # "blipv2_t5", 
                    # 'clip',#* accuracy is high in cue-conflict
                    # "blipv2_7b", #* only 7 percent?
                    # 'blip2', 'blip',
                    # "instructblip_7b", "instructblip_13b", "instructblip_t5", 


                    # "llava_1_6_vicuna_7b",
                    # "llava_1_6_vicuna_13b",
                    # "llava_1_6_34b",
                    # "llava_1_5_7b",
                    # "llava_1_5_13b",
                    # "moe_llava_stablelm",
                    # "moe_llava_qwen",
                    # "moe_llava_phi2",
                    # "emu2_chat",
                    # "qwen_vl_plus",
                    # "instructblip_vicuna_7b",
                    # "instructblip_flan_t5_xl",
                    # "qwen_vl_max",
                    # "gemini_pro_vision",
                    # "internvl_chat",
                    # "llava_rlhf_1_5_7b",
                    # "llava_rlhf_1_5_13b",
                    # "internvl_chat_1_2_plus",
                    # "openai_vision"
                    
                    # # humans
                    # '4_6_years_old',
                    # '7_9_years_old',
                    # '10_12_years_old',
                    # '13_15_years_old',
                    # 'adults',

                    #! SimCLR
                    # 'resnet50_SimCLR_baseline','resnet50_SimCLR_DVD_B',

                    #* our mian models on ecoset
                    # 'resnet50_baseline',
                    # 'resnet50_DVD_B',
                    # 'resnet50_DVD_S',  'resnet50_DVD_SS', 
                    # 'resnet50_DVD_P', 'resnet50_DVD_PP',

                    #! adversarial training
                    # 'resnet50_adult_adv_pgd_linf_eps0_001_k10', 
                    # 'resnet50_adult_adv_pgd_linf_eps0_001_k10_seed2', 
                    # 'resnet50_adult_adv_pgd_linf_eps0_001_k10_seed3', 
                    # 'resnet50_adult_adv_pgd_linf_eps0_0039_k10_seed_1', 
                    # 'resnet50_adult_adv_pgd_linf_eps0_01568_k10_seed_1'

                    # SSL + DVD
                    # miniecsoet
                    # 'resnet50_finetuned_DVD_B_SimCLR_miniecoset_ms08_dvd_data_aug_seed1',
                    # 'resnet50_finetuned_adult_SimCLR_miniecoset_ms08_dvd_data_aug_seed1',
                    # ecoset
                    # 'resnet50_finetuned_DVD_B_SimCLR_ecoset_ms08_dvd_data_aug_seed1',
                    # 'resnet50_finetuned_adult_SimCLR_ecoset_ms08_dvd_data_aug_seed1',
                    # imagenet 1k
                    # 'resnet50_finetuned_DVD_B_SimCLR_imagenet_1k_ms08_dvd_data_aug_seed1',
                    # 'resnet50_finetuned_adult_SimCLR_imagenet_1k_dvd_data_aug_seed1',

                    # BYOL< DINO
                    # 'resnet50_byol_lin_eval_DVD_B_miniecoset_ms08_dvd_data_aug_seed1', 'resnet50_dino_lin_eval_DVD_B_miniecoset_ms08_dvd_data_aug_seed1',
                    # 'resnet50_dino_lin_eval_DVD_B_ecoset_ms08_dvd_data_aug_seed1','resnet50_byol_lin_eval_DVD_B_ecoset_ms08_dvd_data_aug_seed1',
                    'resnet50_dino_lin_eval_DVD_B_imagenet_ms08_dvd_data_aug_seed1','resnet50_byol_lin_eval_DVD_B_magenet_ms08_dvd_data_aug_seed1',

                    #* ablation
                    # 'resnet50_DVD_B_blur_only', 
                    # 'resnet50_DVD_B_color_only', 
                    # 'resnet50_DVD_B_contrast_only', 
                    # 'resnet50_DVD_B_blur_color_only', 'resnet50_DVD_B_blur_contrast_only', 'resnet50_DVD_B_color_contrast_only'

                    # # * our main models on imagenet
                    # 'resnet50_baseline_imagenet', 
                    # 'resnet50_DVD_B_imagenet', 
                    # 'resnet50_DVD_S_imagenet', 'resnet50_DVD_SS_imagenet', 
                    # 'resnet50_DVD_P_imagenet', 'resnet50_DVD_PP_imagenet', 


                    # #* our main models on miniecoset
                    # 'resnet50_mpe1_0_alpha0_2_dn150_0'.replace(".", "_"),
                    # 'resnet50_mpe4_0_alpha0_8_dn50_0'.replace(".", "_"),
                    # 'resnet50_mpe2_0_alpha0_1_dn100_0'.replace(".", "_"),
                    # 'resnet50_baseline_miniecoset', 
                    # 'resnet50_DVD_B_miniecoset', 
                    # 'resnet50_DVD_S_miniecoset', 
                    # 'resnet50_DVD_SS_miniecoset', 
                    # 'resnet50_DVD_P_miniecoset',
                    # 'resnet50_DVD_PP_miniecoset',

                    #* generalization across different architectures
                    # 'alexnet_baseline', 
                    # 'alexnet_DVD_B', 
                    # 'vgg16_baseline', 
                    # 'vgg16_DVD_B', 
                    # 'mobilenet_v3_small_baseline', 
                    # 'mobilenet_v3_small_DVD_B', 
                    # 'shufflenet_v2_x1_0_baseline', 
                    # 'shufflenet_v2_x1_0_DVD_B', 
                    # 'resnet18_baseline', 
                    # 'resnet18_DVD_B',
                    # 'resnet101_baseline', 
                    # 'resnet101_DVD_B', 
                    # 'resnext50_32x4d_baseline', 
                    # 'resnext50_32x4d_DVD_B', 
                    # 'deit_base_patch16_224_baseline', 
                    # 'deit_base_patch16_224_DVD_B', 
                    # 'vit_b_16_baseline', 
                    # 'vit_b_16_DVD_B'

                    #! if chance level, then usually you need to check whether the model path is '' or None
            ]
    #* sweep miniecoset
    # months_per_epochs = [1.0, 2.0, 4.0, 8.0]
    # contrast_thresh = [0.1, 0.2, 0.4, 0.8]  # alpha
    # contrast_spd = [50.0, 100.0, 150.0]     # beta
    # for mpe, alpha, beta in product(months_per_epochs, contrast_thresh, contrast_spd):
    #     model_func_name = f"resnet50_mpe{mpe}_alpha{alpha}_dn{beta}".replace(".", "_")
    #     models.append(model_func_name)


    dataset_trained_on = ["ecoset", "imagenet",'miniecoset_zejin',][1] #c.DEFAULT_DATASETS # ["ecoset"] #or e.g. ["ecoset", "imagenet"]
    #TODO debug other default dataset,'imagenet_validation', 'ecoset_testset', 'miniecoset_zejin_testset'
    datasets =  ["edge", "silhouette", "cue-conflict"][-1:]  # DFAULT+ ["sketch", "stylized"] #c.DEFAULT_DATASETS # ["cue-conflict"] #or e.g. ["cue-conflict", "uniform-noise"]
    params = {"batch_size": 10, "print_predictions": True, "num_workers": 20 , 
            "dataset_trained_on": dataset_trained_on ,
            #* only do 0-1 for our own trained model
            "data_normalization_type": "0-1", #* Preprocessing normalization in 0-1 range, default is mean-std
            'verbose': True,
            }
    Evaluate()(models, datasets, **params)


def run_plotting():
    plot_types =   [ "shape-bias"] #c.DEFAULT_PLOT_TYPES #["accuracy", "shape-bias"] # or e.g. ["accuracy", "shape-bias"]
    plotting_def = plotting_definition_template
    figure_dirname = "NMI/shape_bias/vs_SSL_25th_NOV"  # Directory name to save figures under results/plots/
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)

    # In examples/plotting_definition.py, you can edit
    # plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.


if __name__ == "__main__":
    # 1. evaluate models on out-of-distribution datasets
    run_evaluation()
    # 2. plot the evaluation results
    run_plotting()
