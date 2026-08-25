import copy
import logging
import os
from os.path import join as pjoin
from multiprocessing import Process

import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

from neuroai.plotting import analyses as a
from neuroai.helper import plotting_helper as ph
from neuroai.datasets.experiments import get_experiments
from neuroai import constants as consts
from neuroai.plotting.colors import *
from neuroai.plotting.decision_makers import DecisionMaker
from collections import defaultdict

# global default boundary settings for thin gray transparent
# boundaries to avoid not being able to see the difference
# between two partially overlapping datapoints of the same color:
PLOTTING_EDGE_COLOR = (0.3, 0.3, 0.3, 0.3)
PLOTTING_EDGE_WIDTH = 0.02


from neuroai import Evaluate



def run_evaluation(model_name, load_checkpoint_path =None, dataset_trained_on = 'ecoset', data_normalization_type = "0-1" , image_size = None):

    models = [model_name] # only 1 model here
    datasets =  ["cue-conflict"][-1:]  # consts.DEFAULT_DATASETS, here only shape bias
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 10 , 
            'load_checkpoint_path':load_checkpoint_path, # Whether load from desired weight path
            "dataset_trained_on": dataset_trained_on, # e.g. ["ecoset", "imagenet"] or  consts.DEFAULT_DATASETS 
            "data_normalization_type": data_normalization_type, #* Preprocessing normalization in 0-1 range, default is mean-std
            'image_size': image_size, #* image size for the model
            'verbose': False,
            }
    Evaluate()(models, datasets, **params)


def get_shape_bias(
                    model_name = None,
                    load_checkpoint_path = None,
                    evaluation = False,
                    dataset_trained_on = 'ecoset',
                    data_normalization_type = "0-1",
                    image_size = None,
                    dataset_name ="cue-conflict",
                    analysis=a.ShapeBias(),
                    matrixplot_savedir=None,
                    order_by='humans'):
    
    if evaluation:
        run_evaluation(model_name, load_checkpoint_path=load_checkpoint_path, dataset_trained_on = dataset_trained_on, data_normalization_type = data_normalization_type, image_size=image_size)

    def decision_maker_fun(df):
        """Decision makers to compare a few models with human observers.
        For each model, define:
        - a color using rgb(42, 42, 42)
        - a plotting symbol by setting marker;
        a list of markers can be found here:
        https://matplotlib.org/3.1.0/api/markers_api.html
        """

        decision_makers = []
        # Testing model
        decision_makers.append(DecisionMaker(name_pattern=model_name,
                            color=rgb(0, 0, 0), marker="o", df=df,
                            plotting_name= model_name))
        # No need Human
        # decision_makers.append(DecisionMaker(name_pattern="subject-*",
        #                     color=rgb(165, 30, 55), marker="D", df=df,
        #                     plotting_name="humans"))

        return decision_makers

    assert dataset_name == "cue-conflict"

    ds = get_experiments([dataset_name])[0]
    df = ph.get_experimental_data(ds)

    fontsize = 25
    ticklength = 10
    markersize = 250

    classes = df["category"].unique()
    num_classes = len(classes)

    # No plot setup

    # icons besides y axis
    ## determine order of icons
    for dmaker in decision_maker_fun(df):
        if dmaker.plotting_name == order_by:
            df_selection = df.loc[(df["subj"].isin(dmaker.decision_makers))]
            class_avgs = []
            for cl in classes:
                df_class_selection = df_selection.query("category == '{}'".format(cl))
                class_avgs.append(1 - analysis.analysis(df=df_class_selection)['shape-bias'])
            sorted_indices = np.argsort(class_avgs)
            classes = classes[sorted_indices]
            break

    # icon placement is calculated in axis coordinates
    WIDTH = 1 / num_classes  #
    XPOS = -1.25 * WIDTH  # placement left of yaxis (-WIDTH) plus some spacing (-.25*WIDTH)
    YPOS = -0.5
    HEIGHT = 1
    MARGINX = 1 / 10 * WIDTH  # vertical whitespace between icons
    MARGINY = 1 / 10 * HEIGHT  # horizontal whitespace between icons

    left = XPOS + MARGINX
    right = XPOS + WIDTH - MARGINX

    for i in range(num_classes):
        bottom = i + MARGINY + YPOS
        top = (i + 1) - MARGINY + YPOS
        iconpath = pjoin(consts.ICONS_DIR, "{}.png".format(classes[i]))
        plt.imshow(plt.imread(iconpath), extent=[left, right, bottom, top], aspect='auto', clip_on=False)

    # plot horizontal intersection lines
    for i in range(num_classes - 1):
        plt.plot([0, 1], [i + .5, i + .5], c='gray', linestyle='dotted', alpha=0.4)

    results = []
    avg_shape_bias_dict = defaultdict(list)
    avg_shape_bias_dict["classes"] = classes # Just saving the categories names
    shape_acc_dict = defaultdict(list)
    texture_acc_dict = defaultdict(list)
    
    # plot average shapebias + scatter points
    for dmaker in decision_maker_fun(df):
        df_selection = df.loc[(df["subj"].isin(dmaker.decision_makers))]
        result_df = analysis.analysis(df=df_selection)
        avg = 1 - result_df['shape-bias']
        # ax.plot([avg, avg], [-1, num_classes], color=dmaker.color)

        class_avgs = []
        class_shape_accs = []
        class_texture_accs = []
        for cl in classes:
            df_class_selection = df_selection.query("category == '{}'".format(cl))
            class_avgs.append(1 - analysis.analysis(df=df_class_selection)['shape-bias'])

            class_shape_accs.append(analysis.analysis(df=df_class_selection)['fraction-correct-shape'])
            class_texture_accs.append(analysis.analysis(df=df_class_selection)['fraction-correct-texture'])

        # No plot setup for scatter

        # import pdb; pdb.set_trace()
        #* print shape bias across categories
        avg_shape_bias = 1-avg
        median_shape_bias = 1-np.median(class_avgs)
        results.append({
            'plotting_name': dmaker.plotting_name,
            'plotting_color':dmaker.color,
            'shape_bias': avg_shape_bias,
            'median_shape_bias': median_shape_bias,
        })

        avg_shape_bias_dict[dmaker.plotting_name] = 1 - np.array(class_avgs) # shape bias  = 1- texture bais
        shape_acc_dict[dmaker.plotting_name]  = class_shape_accs
        texture_acc_dict[dmaker.plotting_name]  = class_texture_accs

    shape_bias_across = avg_shape_bias_dict[model_name]
    shape_accs = shape_acc_dict[model_name]
    texture_accs = texture_acc_dict[model_name]
    return avg_shape_bias, shape_bias_across, classes, shape_accs, texture_accs  if model_name else avg_shape_bias_dict


if __name__ == '__main__':
    model_name = 'resnet50_DVD-B'
    evaluation = True
    load_checkpoint_path = None
    dataset_trained_on = ['ecoset', 'imagenet'][0]
    data_normalization_type = ["0-1", "mean-std"][0] #* #* Preprocessing normalization in 0-1 range, default is mean-std
    matrixplot_savedir = None
    shape_bias, shape_bias_across, classes = get_shape_bias(model_name, evaluation = evaluation, load_checkpoint_path=load_checkpoint_path,dataset_trained_on = dataset_trained_on, data_normalization_type = data_normalization_type, matrixplot_savedir=matrixplot_savedir)
    import pdb; pdb.set_trace()


