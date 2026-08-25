# /!usr/bin/env python3

"""
Define decision makers (either human participants or CNN models).
"""

import fnmatch

from neuroai import constants as c
from neuroai.plotting.colors import *
from neuroai.plotting.decision_makers import DecisionMaker


def _has_subject_match(df, name_pattern):
    patterns = [name_pattern] if isinstance(name_pattern, str) else list(name_pattern)
    subjects = df.subj.unique()
    return any(fnmatch.fnmatch(subj, pattern) for subj in subjects for pattern in patterns)


def _append_decision_maker_if_present(decision_makers, df, *, name_pattern,
                                      color, marker, plotting_name):
    if _has_subject_match(df, name_pattern):
        decision_makers.append(
            DecisionMaker(
                name_pattern=name_pattern,
                color=color,
                marker=marker,
                df=df,
                plotting_name=plotting_name,
            )
        )


def _append_default_human_decision_makers(decision_makers, df):
    cohort_specs = [
        ("4_6_years_old", "humans(4-6 y/o)"),
        ("7_9_years_old", "humans(7-9 y/o)"),
        ("10_12_years_old", "humans(10-12 y/o)"),
        ("13_15_years_old", "humans(13-15 y/o)"),
        ("adults", "humans(adults)"),
    ]

    for cohort_name, plotting_name in cohort_specs:
        _append_decision_maker_if_present(
            decision_makers,
            df,
            name_pattern=cohort_name,
            color=rgb(207, 138, 153),
            marker="D",
            plotting_name=plotting_name,
        )

    _append_decision_maker_if_present(
        decision_makers,
        df,
        name_pattern="subject-*",
        color=rgb(165, 30, 55),
        marker="D",
        plotting_name="humans",
    )


def plotting_definition_template(df):
    """Decision makers to compare a few models with human observers.

    This exemplary definition can be adapted for the
    desired purpose, e.g. by adding more/different models.

    Note that models will need to be evaluated first, before
    their data can be plotted.

    For each model, define:
    - a color using rgb(42, 42, 42)
    - a plotting symbol by setting marker;
      a list of markers can be found here:
      https://matplotlib.org/3.1.0/api/markers_api.html
    """

    decision_makers = []

    #* Default models
    # decision_makers = get_comparison_decision_makers(df,include_humans=False)

    #* Sweep across miniecoset
    model_color_map = {
        # "resnet50_mpe1.0_alpha0.2_dn150.0": rgb(112, 207, 87), #!
        # # "resnet50_mpe1.0_alpha0.2_dn100.0": rgb(68, 191, 112),
        # # "resnet50_mpe1.0_alpha0.8_dn150.0": rgb(49, 180, 123),
        # # "resnet50_mpe1.0_alpha0.4_dn50.0": rgb(61, 187, 116),
        # # "resnet50_mpe1.0_alpha0.4_dn100.0": rgb(36, 170, 130),
        # # "resnet50_mpe1.0_alpha0.4_dn150.0": rgb(56, 185, 119),
        # # "resnet50_mpe1.0_alpha0.1_dn100.0": rgb(31, 152, 139),
        # # "resnet50_mpe1.0_alpha0.8_dn50.0": rgb(30, 157, 137),
        # # "resnet50_mpe1.0_alpha0.1_dn150.0": rgb(32, 164, 134),
        # # "resnet50_mpe1.0_alpha0.2_dn50.0": rgb(30, 157, 137),
        # # "resnet50_mpe1.0_alpha0.8_dn100.0": rgb(30, 155, 138),
        # # "resnet50_mpe2.0_alpha0.2_dn150.0": rgb(48, 105, 142),
        # # "resnet50_mpe1.0_alpha0.1_dn50.0": rgb(30, 157, 137),
        # # "resnet50_mpe2.0_alpha0.4_dn150.0": rgb(38, 130, 142),
        # "resnet50_mpe2.0_alpha0.1_dn100.0": rgb(38, 129, 142),#!
        # # "resnet50_mpe2.0_alpha0.2_dn100.0": rgb(30, 153, 138),
        # # "resnet50_mpe2.0_alpha0.8_dn50.0": rgb(35, 137, 141),
        # # "resnet50_mpe2.0_alpha0.4_dn100.0": rgb(39, 128, 142),
        # # "resnet50_mpe2.0_alpha0.1_dn150.0": rgb(36, 135, 141),
        # # "resnet50_mpe2.0_alpha0.1_dn50.0": rgb(46, 110, 142),
        # # "resnet50_mpe2.0_alpha0.4_dn50.0": rgb(41, 122, 142),
        # # "resnet50_mpe4.0_alpha0.1_dn150.0": rgb(47, 107, 142),
        # # "resnet50_mpe2.0_alpha0.8_dn150.0": rgb(39, 125, 142),
        # # "resnet50_mpe2.0_alpha0.8_dn100.0": rgb(39, 125, 142),
        # # "resnet50_mpe4.0_alpha0.4_dn50.0": rgb(48, 106, 142),
        # # "resnet50_mpe4.0_alpha0.4_dn150.0": rgb(51, 98, 141),
        # # "resnet50_mpe4.0_alpha0.1_dn100.0": rgb(64, 70, 136),
        # # "resnet50_mpe2.0_alpha0.2_dn50.0": rgb(53, 92, 141),
        # # "resnet50_mpe4.0_alpha0.2_dn100.0": rgb(63, 72, 137),
        # # "resnet50_mpe4.0_alpha0.8_dn150.0": rgb(65, 67, 135),
        # # "resnet50_mpe4.0_alpha0.8_dn100.0": rgb(72, 20, 102),
        # "resnet50_mpe4.0_alpha0.8_dn50.0": rgb(67, 61, 132), #!
        # 'resnet50_DVD_P_miniecoset': rgb(67, 61, 132), #!
        # # "resnet50_mpe4.0_alpha0.2_dn50.0": rgb(68, 1, 84),
        # # "resnet50_mpe4.0_alpha0.1_dn50.0": rgb(72, 28, 110),
        # # "resnet50_mpe4.0_alpha0.2_dn150.0": rgb(71, 44, 122),
        # # "resnet50_mpe4.0_alpha0.4_dn100.0": rgb(72, 20, 102),
        # # "resnet50_mpe8.0_alpha0.1_dn50.0": rgb(71, 45, 123),
        # # "resnet50_mpe8.0_alpha0.2_dn150.0": rgb(72, 32, 113),
        # # "resnet50_mpe8.0_alpha0.8_dn100.0": rgb(72, 38, 119),
        # # "resnet50_mpe8.0_alpha0.2_dn50.0": rgb(72, 25, 107),
        # # "resnet50_mpe8.0_alpha0.4_dn150.0": rgb(72, 20, 102),
        # # "resnet50_mpe8.0_alpha0.1_dn150.0": rgb(70, 6, 90),
        # # "resnet50_mpe8.0_alpha0.2_dn100.0": rgb(69, 5, 88),
        # # "resnet50_mpe8.0_alpha0.8_dn50.0": rgb(68, 1, 84),
        # # "resnet50_mpe8.0_alpha0.4_dn50.0": rgb(68, 1, 84),
        # # "resnet50_mpe8.0_alpha0.4_dn100.0": rgb(68, 1, 84),
        # # "resnet50_mpe8.0_alpha0.1_dn100.0": rgb(68, 1, 84),
        # # "resnet50_mpe8.0_alpha0.8_dn150.0": rgb(68, 1, 84),
        # "resnet50_baseline_miniecoset": rgb(0, 0, 0),
    }

    # Iterate over the model_color_map to create DecisionMaker instances
    for model_key, color in reversed(model_color_map.items()):
        model_name = f"{model_key}"
        # if model_name in  ['resnet50_baseline_miniecoset','resnet50_mpe2.0_alpha0.1_dn100.0', 'resnet50_mpe1.0_alpha0.2_dn150.0', 'resnet50_mpe4.0_alpha0.8_dn50.0']:
        r, g, b = [int(255 * c) for c in color]
        decision_makers.append(
            DecisionMaker(
                name_pattern=model_name.replace(".", "_"),
                color=rgb(r, g, b),
                marker="o",
                df=df,
                #! define whehter only plot representative models
                plotting_name=model_name.replace(".", "_"), # if model_name in  ['resnet50_baseline_miniecoset','resnet50_mpe2.0_alpha0.1_dn100.0', 'resnet50_mpe1.0_alpha0.2_dn150.0', 'resnet50_mpe4.0_alpha0.8_dn50.0'] else ''
            )
        )
    
    #* Plot across datasets 
    for dataset_suffix in ["_miniecoset","", "_imagenet", ][:]:
        # Baseline RN50 on ecoset
        name = f"resnet50_baseline{dataset_suffix}"
        plot_name = f"resnet50 baseline {dataset_suffix.lstrip('_')}"
        decision_makers.append(DecisionMaker(name_pattern=name,
                            color=rgb(0, 0, 0), marker="o", df=df,
                            plotting_name=plot_name))
                            
        # # Performance foavoer
        # for idx, suffix in enumerate(["PP", "P" ][1:]):
        #     name = f"resnet50_DVD_{suffix}{dataset_suffix}"
        #     plot_name = f"DVD-{suffix} {dataset_suffix.lstrip('_')}"
        #     decision_makers.append(DecisionMaker(name_pattern=name, 
        #                     color=[deep_blue, deep_blue][idx], marker="o", df=df, 
        #                     plotting_name=plot_name))
        
        # Balanced
        for suffix in ["B"]:
            name = f"resnet50_DVD_{suffix}{dataset_suffix}"
            plot_name = f"DVD-{suffix} {dataset_suffix.lstrip('_')}"
            decision_makers.append(DecisionMaker(name_pattern=name, 
                            color= teal_green, marker="o", df=df, 
                            plotting_name=plot_name))
                            
        # Shape bias favored
        # for suffix in ["S", "SS"][:]:
        #     if dataset_suffix == "_imagenet" and suffix == 'SS':
        #         # pass
        #         suffix = "S"
        #         break
        #     # else:
        #     name = f"resnet50_DVD_{suffix}{dataset_suffix}"
        #     plot_name = f"DVD-{suffix} {dataset_suffix.lstrip('_')}"
        #     decision_makers.append(DecisionMaker(name_pattern=name, 
        #                     color=bright_green, marker="o", df=df, 
        #                     plotting_name=plot_name))

    #* Plot across architectures
    # for architecture_suffix in [
    #                             # 'resnet50_SimCLR', #* SimCLR
    #                             'alexnet',
    #                             'vgg16',
    #                             'resnet101',
    #                             'mobilenet_v3_small',
    #                             # 'shufflenet_v2_x1_0',
    #                             'resnet18',
    #                             'resnext50_32x4d',
    #                             'deit_base_patch16_224',
    #                             'vit_b_16',
    #                         ]:
    #     # Baseline
    #     if architecture_suffix in ['vgg16', 'resnet101', 'mobilenet_v3_small', 'shufflenet_v2_x1_0','resnet18'][:0]: #-1
    #         pass
    #     else:
    #         name = f"{architecture_suffix}_baseline"
    #         decision_makers.append(DecisionMaker(name_pattern=name,
    #                             color=rgb(0, 0, 0), marker="o", df=df,
    #                             plotting_name=name[:-9]+' baseline'))
        
    #     # Balanced
    #     if architecture_suffix in ['resnet18', 'vit_b_16',][:0]:
    #         pass
    #     else:
    #         name = f"{architecture_suffix}_DVD_B"
    #         decision_makers.append(DecisionMaker(name_pattern=name,
    #                             color=teal_green, marker="o", df=df,
    #                             plotting_name=name[:-6]+' DVD-B'))

    #* Plot ablation studies
    # ablation_colors = {
    #     'resnet50_DVD_B_blur_only': rgb(100, 200, 100),           # lighter green
    #     'resnet50_DVD_B_color_only': rgb(50, 180, 130),           # green-teal
    #     'resnet50_DVD_B_contrast_only': rgb(0, 150, 0),           # dark green
    #     'resnet50_DVD_B_blur_color_only': rgb(80, 170, 90),       # medium green
    #     'resnet50_DVD_B_blur_contrast_only': rgb(30, 160, 60),    # forest green
    #     'resnet50_DVD_B_color_contrast_only': rgb(60, 160, 110),  # sea green
    # }


    # for ablation_condition in [
    #     'resnet50_DVD_B_blur_only',
    #     'resnet50_DVD_B_color_only',
    #     'resnet50_DVD_B_contrast_only',
    #     'resnet50_DVD_B_blur_color_only',
    #     'resnet50_DVD_B_blur_contrast_only',
    #     'resnet50_DVD_B_color_contrast_only',
    # ]:
    #     if df[df['subj'] == ablation_condition].empty:
    #         print(f"Skipping {ablation_condition} because no matching data was found.")
    #         continue

    #     plot_name = ablation_condition.replace("resnet50_", "").replace("_", " ")
    #     decision_makers.append(
    #         DecisionMaker(
    #             name_pattern=ablation_condition,
    #             color=ablation_colors[ablation_condition],
    #             marker="o",
    #             df=df,
    #             plotting_name=plot_name,
    #         )
    #     )

    # #! #! adversarial training
    # for adv_condition in ['resnet50_adult_adv_pgd_linf_eps0_001_k10', 'resnet50_adult_adv_pgd_linf_eps0_001_k10_seed2', 'resnet50_adult_adv_pgd_linf_eps0_001_k10_seed3', 'resnet50_adult_adv_pgd_linf_eps0_0039_k10_seed_1', 'resnet50_adult_adv_pgd_linf_eps0_01568_k10_seed_1']:
    #     if df[df['subj'] == adv_condition].empty:
    #         print(f"Skipping {adv_condition} because no matching data was found.")
    #         continue

    #     plot_name = adv_condition.replace("resnet50_", "").replace("_", " ")
    #     decision_makers.append(
    #         DecisionMaker(
    #             name_pattern=adv_condition,
    #             color=orange1,
    #             marker="o",
    #             df=df,
    #             plotting_name=plot_name,
    #         )
    #     )

    #! SSL
    #! #! SSL models
    for ssl_condition in [
        # miniecoset
        "resnet50_finetuned_DVD_B_SimCLR_miniecoset_ms08_dvd_data_aug_seed1",
        # "resnet50_finetuned_adult_SimCLR_miniecoset_ms08_dvd_data_aug_seed1",

        # ecoset
        "resnet50_finetuned_DVD_B_SimCLR_ecoset_ms08_dvd_data_aug_seed1",
        # "resnet50_finetuned_adult_SimCLR_ecoset_ms08_dvd_data_aug_seed1",

        # imagenet 1k
        "resnet50_finetuned_DVD_B_SimCLR_imagenet_1k_ms08_dvd_data_aug_seed1",
        # "resnet50_finetuned_adult_SimCLR_imagenet_1k_dvd_data_aug_seed1",

        'resnet50_byol_lin_eval_DVD_B_miniecoset_ms08_dvd_data_aug_seed1', 
        'resnet50_byol_lin_eval_DVD_B_ecoset_ms08_dvd_data_aug_seed1',
        'resnet50_byol_lin_eval_DVD_B_magenet_ms08_dvd_data_aug_seed1',
        
        'resnet50_dino_lin_eval_DVD_B_miniecoset_ms08_dvd_data_aug_seed1',
        'resnet50_dino_lin_eval_DVD_B_ecoset_ms08_dvd_data_aug_seed1',
        'resnet50_dino_lin_eval_DVD_B_imagenet_ms08_dvd_data_aug_seed1',
    ]:
        if df[df["subj"] == ssl_condition].empty:
            print(f"Skipping {ssl_condition} because no matching data was found.")
            continue

        plot_name = ssl_condition.replace("resnet50_", "").replace("_", " ")

        decision_makers.append(
            DecisionMaker(
                name_pattern=ssl_condition,
                color='lightblue',
                marker="o",
                df=df,
                plotting_name=plot_name,
            )
        )

    
    

    _append_default_human_decision_makers(decision_makers, df)

    return decision_makers


def get_comparison_decision_makers(df, include_humans=True,
                                   humans_last=True):
    """Decision makers used in our paper."""

    d = []
    not_needed_models = [
                    ]

    # # 1. supervised models
    # for model in c.TORCHVISION_MODELS:
    #     if model not in not_needed_models:
    #         d.append(DecisionMaker(name_pattern=model,
    #                             color=rgb(230, 230, 230), df=df, # grey1
    #                             plotting_name=model))

    # # 2. self-supervised models
    # d.append(DecisionMaker(name_pattern="simclr_resnet50x1",
    #                        color= grey2, marker="o", df=df, # orange2
    #                        plotting_name="SimCLR: ResNet-50"))
    # for model in c.PYCONTRAST_MODELS:
    #     if model not in not_needed_models:
    #         d.append(DecisionMaker(name_pattern=model,
    #                             color=grey2, marker="o", df=df,
    #                             plotting_name=model+": ResNet-50"))



    # 3. adversarially robust models --> unnatural, 
    # d += [DecisionMaker(name_pattern="resnet50_l2_eps0",
    #                     color=rgb(196, 205, 229), marker="o", df=df,
    #                     plotting_name="ResNet-50 L2 eps 0.0"),
    #       DecisionMaker(name_pattern="resnet50_l2_eps0_5",
    #                     color=rgb(176, 190, 220), marker="o", df=df,
    #                     plotting_name="ResNet-50 L2 eps 0.5"),
    #       DecisionMaker(name_pattern="resnet50_l2_eps1",
    #                     color=rgb(134, 159, 203), marker="o", df=df,
    #                     plotting_name="ResNet-50 L2 eps 1.0"),
    #       DecisionMaker(name_pattern="resnet50_l2_eps3",
    #                     color=rgb(86, 130, 186), marker="o", df=df,
    #                     plotting_name="ResNet-50 L2 eps 3.0"),
    #       DecisionMaker(name_pattern="resnet50_l2_eps5",
    #                     color=blue2, marker="o", df=df,
    #                     plotting_name="ResNet-50 L2 eps 5.0")]

    # 4. vision transformers without large-scale pretraining
    # d.append(DecisionMaker(name_pattern="vit_small_patch16_224",
    #                        color=grey3, marker="v", df=df, # rgb(144, 159, 110)
    #                        plotting_name="ViT-S"))
    # d.append(DecisionMaker(name_pattern="vit_base_patch16_224",
    #                        color=grey3, marker="v", df=df,
    #                        plotting_name="ViT-B"))
    # d.append(DecisionMaker(name_pattern="vit_large_patch16_224",
    #                        color=grey3, marker="v", df=df,
    #                        plotting_name="ViT-L"))

    if not humans_last:
        if include_humans:
            d.append(DecisionMaker(name_pattern="subject-*",
                                   color=red, marker="D", df=df,
                                   plotting_name="humans(adult)"))
        d.append(DecisionMaker(name_pattern="clip",
                               color=brown1, marker="v", df=df,
                               plotting_name="CLIP: ViT-B (400M)"))
    
    # One SWSL is enough: 0.6-0.7 shape bias
    # # d.append(DecisionMaker(name_pattern="ResNeXt101_32x16d_swsl",
    # #                        color=purple1, marker="o", df=df,
    # #                        plotting_name="SWSL: ResNeXt-101 (940M)")) 
    # d.append(DecisionMaker(name_pattern="resnet50_swsl",
    #                        color=grey4, marker="o", df=df, # purple1
    #                        plotting_name="SWSL: ResNet-50 (940M)"))
 
    # bitm_col = rgb(153, 142, 195) 
    # d.append(DecisionMaker(name_pattern="BiTM_resnetv2_152x4",
    #                        color=grey4, marker="o", df=df, # bitm_col
    #                        plotting_name="BiT-M: ResNet-152x4 (14M)"))
    # d.append(DecisionMaker(name_pattern="BiTM_resnetv2_101x1",
    #                        color=grey4, marker="o", df=df,
    #                        plotting_name="BiT-M: ResNet-101x1 (14M)"))
    # d.append(DecisionMaker(name_pattern="BiTM_resnetv2_50x1",
    #                        color=grey4, marker="o", df=df,
    #                        plotting_name="BiT-M: ResNet-50x1 (14M)"))
                           
    # d.append(DecisionMaker(name_pattern="BiTM_resnetv2_152x2",
    #                        color=bitm_col, marker="o", df=df,
    #                        plotting_name="BiT-M: ResNet-152x2 (14M)"))
    # d.append(DecisionMaker(name_pattern="BiTM_resnetv2_101x3",
    #                        color=bitm_col, marker="o", df=df,
    #                        plotting_name="BiT-M: ResNet-101x3 (14M)"))
    # d.append(DecisionMaker(name_pattern="BiTM_resnetv2_50x3",
    #                        color=bitm_col, marker="o", df=df,
    #                        plotting_name="BiT-M: ResNet-50x3 (14M)"))

    # d.append(DecisionMaker(name_pattern="transformer_L16_IN21K",
    #                        color=green1, marker="v", df=df,
    #                        plotting_name="ViT-L (14M)"))
    # d.append(DecisionMaker(name_pattern="transformer_B16_IN21K",
    #                        color=green1, marker="v", df=df,
    #                        plotting_name="ViT-B (14M)"))

    # d.append(DecisionMaker(name_pattern="efficientnet_l2_noisy_student_475",
    #                        color=metallic, marker="o", df=df,
    #                        plotting_name="Noisy Student: ENetL2 (300M)"))
    
    #* multi modality
    # clipRN50
    # d.append(DecisionMaker(name_pattern="clipRN50",
    #                         color=grey5, marker="v", df=df, # brown1
    #                         plotting_name="CLIP: ResNet50"))
    # d.append(DecisionMaker(name_pattern="clipRN50_CCM_12m",
    #                         color=grey5, marker="v", df=df, # brown1
    #                         plotting_name="CLIP-RN50 (12m)"))
    # # clipRN50_CCM_12m_DVD
    # d.append(DecisionMaker(name_pattern="clipRN50_CCM_12m_DVD",
    #                         color=grey5, marker="v", df=df, # brown1
    #                         plotting_name="CLIP-RN50 (12m) + DVD"))
    # clipRN50_CCM_12m_all_epochs
    # d.append(DecisionMaker(name_pattern="clipRN50_CCM_12m_all_epochs",
    #                         color=grey5, marker="v", df=df, # brown1
    #                         plotting_name="CLIP-RN50(12m) all epochs"))

    # d.append(DecisionMaker(name_pattern="clip",
    #                         color=grey5, marker="v", df=df, # brown1
    #                         plotting_name="CLIP: ViT-B (400M)"))

    #* VLMs 
    # model_id_to_name = {
    #                     "llava_1_6_vicuna_7b": "LLaVA-NeXT 7B",
    #                     "llava_1_6_vicuna_13b": "LLaVA-NeXT 13B",
    #                     "llava_1_6_34b": "LLaVA-NeXT 34B",
    #                     "llava_1_5_7b": "LLaVA v1.5 7B",
    #                     "llava_1_5_13b": "LLaVA v1.5 13B",
    #                     "moe_llava_stablelm": "MoE-LLaVA-StableLM",
    #                     "moe_llava_qwen": "MoE-LLaVA-Qwen",
    #                     "moe_llava_phi2": "MoE-LLaVA-Phi2",
    #                     "emu2_chat": "Emu2-Chat",
    #                     "qwen_vl_plus": "Qwen-VL Plus",
    #                     "instructblip_vicuna_7b": "InstructBLIP Vicuna-7B",
    #                     "instructblip_flan_t5_xl": "InstructBLIP Flan-T5-xl",
    #                     "qwen_vl_max": "Qwen-VL Max",
    #                     "gemini_pro_vision": "Gemini Pro Vision",
    #                     "internvl_chat": "InternVL Chat 1.1",
    #                     "llava_rlhf_1_5_7b": "LLaVA-RLHF 7B",
    #                     "llava_rlhf_1_5_13b": "LLaVA-RLHF 13B",
    #                     "internvl_chat_1_2_plus": "InternVL Chat 1.2+",
    #                     "openai_vision": "GPT-4V"
    #                 }


    # d.append(DecisionMaker(name_pattern="llava_1_5_7b",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="LLaVA v1.5 7B"))
    # d.append(DecisionMaker(name_pattern="llava_1_5_13b",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="LLaVA v1.5 13B"))
    # d.append(DecisionMaker(name_pattern="llava_1_6_vicuna_7b",
    #                    color=grey7, marker="v", df=df,
    #                    plotting_name="LLaVA-NeXT 7B"))
    # d.append(DecisionMaker(name_pattern="llava_1_6_vicuna_13b",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="LLaVA-NeXT 13B"))
    # d.append(DecisionMaker(name_pattern="llava_1_6_34b",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="LLaVA-NeXT 34B"))


    # d.append(DecisionMaker(name_pattern="llava_rlhf_1_5_7b",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="LLaVA-RLHF 7B"))

    # d.append(DecisionMaker(name_pattern="llava_rlhf_1_5_13b",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="LLaVA-RLHF 13B"))

    # d.append(DecisionMaker(name_pattern="moe_llava_stablelm",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="MoE-LLaVA-StableLM"))
    # d.append(DecisionMaker(name_pattern="moe_llava_qwen",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="MoE-LLaVA-Qwen"))
    # d.append(DecisionMaker(name_pattern="moe_llava_phi2",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="MoE-LLaVA-Phi2"))

    # d.append(DecisionMaker(name_pattern="qwen_vl_plus",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="Qwen-VL Plus"))
    # d.append(DecisionMaker(name_pattern="qwen_vl_max",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="Qwen-VL Max"))

    # d.append(DecisionMaker(name_pattern="emu2_chat",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="Emu2-Chat"))

    # d.append(DecisionMaker(name_pattern="instructblip_vicuna_7b",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="InstructBLIP Vicuna-7B"))
    # d.append(DecisionMaker(name_pattern="instructblip_flan_t5_xl",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="InstructBLIP Flan-T5-xl"))

    # d.append(DecisionMaker(name_pattern="internvl_chat",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="InternVL Chat 1.1"))
    # d.append(DecisionMaker(name_pattern="internvl_chat_1_2_plus",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="InternVL Chat 1.2+"))

    # d.append(DecisionMaker(name_pattern="gemini_pro_vision",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="Gemini Pro Vision"))
    # d.append(DecisionMaker(name_pattern="openai_vision",
    #                     color=grey7, marker="v", df=df,
    #                     plotting_name="ChatGPT-4V"))

    if humans_last:
        if include_humans:


            d.append(DecisionMaker(name_pattern="subject-*",
                                   color='green', marker="D", df=df, # red
                                   plotting_name="humans"))


    return d
