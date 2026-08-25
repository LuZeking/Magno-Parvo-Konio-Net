#!/usr/bin/env python

import numpy as np
from abc import ABC, abstractmethod

from ..helper import human_categories as hc
from ..helper import wordnet_functions as wnf


class DecisionMapping(ABC):
    def check_input(self, probabilities):
        assert type(probabilities) is np.ndarray
        assert (probabilities >= 0.0).all() and (probabilities <= 1.0).all()

    @abstractmethod
    def __call__(self, probabilities):
        pass

# All categories
class ImageNetProbabilitiesTo1000ClassesMapping(DecisionMapping):
    """Return the WNIDs sorted by probabilities."""
    def __init__(self):
        self.categories = wnf.get_ilsvrc2012_WNIDs()

    def __call__(self, probabilities):
        self.check_input(probabilities)
        sorted_indices = np.flip(np.argsort(probabilities), axis=-1)
        return np.take(self.categories, sorted_indices, axis=-1)

class ecosetProbabilitiesTo565ClassesMapping(DecisionMapping):
    """Return the WNIDs sorted by probabilities."""
    def __init__(self):
        self.categories = ['man', 'house', 'car', 'woman', 'phone', 'bed', 'gun', 'book', 'dog', 'ball', 'fire', 'horse', 'city', 'fish', 'child', 'boat', 'table', 'tree', 'clock', 'bag', 'camera', 'cup', 'ship', 'kitchen', 'key', 'bird', 'pig', 'bus', 'bridge', 'pizza', 'computer', 'church', 'doll', 'bell', 'stairs', 'apple', 'flower', 'ring', 'snake', 'mountain', 'road', 'wall', 'tiger', 'toilet', 'train', 'bottle', 'turtle', 'cookie', 'egg', 'river', 'cat', 'truck', 'basket', 'bear', 'moon', 'milk', 'blanket', 'lemon', 'frog', 'pillow', 'elephant', 'cow', 'banana', 'goat', 'knife', 'ladder', 'popcorn', 'refrigerator', 'jar', 'jail', 'hamburger', 'toast', 'umbrella', 'bean', 'castle', 'flashlight', 'tomato', 'strawberry', 'leopard', 'donkey', 'axe', 'mailbox', 'grape', 'vase', 'carrot', 'tractor', 'cupcake', 'fern', 'bagel', 'telescope', 'cactus', 'tent', 'microscope', 'kite', 'lantern', 'octopus', 'lamp', 'blender', 'burrito', 'mango', 'binoculars', 'steak', 'gravel', 'escalator', 'walrus', 'horseshoe', 'antelope', 'tongs', 'porcupine', 'camcorder', 'mousetrap', 'lion', 'cauliflower', 'shower', 'hotdog', 'warthog', 'guardrail', 'thimble', 'crawfish', 'dustpan', 'eyedropper', 'nectarine', 'flyswatter', 'lollypop', 'lightbulb', 'corn', 'cave', 'pie', 'spider', 'bread', 'motorcycle', 'monkey', 'whale', 'airplane', 'shovel', 'bucket', 'rabbit', 'necklace', 'moose', 'glass', 'drum', 'mop', 'bracelet', 'spoon', 'stove', 'lettuce', 'ashtray', 'lake', 'noodles', 'walnut', 'pastry', 'ferret', 'fig', 'eggplant', 'violin', 'chipmunk', 'milkshake', 'blackberry', 'sushi', 'apricot', 'drawers', 'doughnut', 'lawnmower', 'snowplow', 'chalkboard', 'backpack', 'alligator', 'cockroach', 'lime', 'lighthouse', 'dolphin', 'piano', 'earring', 'chicken', 'blueberry', 'grapefruit', 'thermometer', 'cranberry', 'iceberg', 'wasp', 'tweezers', 'asparagus', 'croissant', 'teapot', 'needle', 'acorn', 'bumblebee', 'anvil', 'seashell', 'birdhouse', 'thyme', 'thumbtack', 'saltshaker', 'paperclip', 'matchstick', 'envelope', 'pineapple', 'wheelchair', 'butterfly', 'camel', 'balloon', 'beach', 'guitar', 'crate', 'wrench', 'eggroll', 'taco', 'rat', 'meatball', 'emerald', 'omelet', 'sheep', 'jukebox', 'desert', 'raspberry', 'snail', 'pistachio', 'groundhog', 'videotape', 'jellyfish', 'carousel', 'hippo', 'pear', 'bullet', 'pliers', 'toothpick', 'stagecoach', 'hourglass', 'coliseum', 'blowtorch', 'treadmill', 'cake', 'bobcat', 'cumin', 'hedgehog', 'opossum', 'pumpkin', 'bowl', 'worm', 'candy', 'clarinet', 'peanut', 'gondola', 'padlock', 'television', 'barnacle', 'leek', 'cashew', 'streetlamp', 'razor', 'peach', 'pudding', 'coin', 'grasshopper', 'pea', 'oven', 'chalk', 'bee', 'bicycle', 'screwdriver', 'pencil', 'jaguar', 'garlic', 'rhino', 'wheat', 'waterfall', 'squirrel', 'bulldozer', 'closet', 'broom', 'ladle', 'starfish', 'scoreboard', 'rice', 'crouton', 'lasagna', 'flea', 'towel', 'bench', 'hammock', 'windmill', 'pan', 'avocado', 'guacamole', 'cane', 'sprinkler', 'microphone', 'outhouse', 'anthill', 'tortilla', 'eggbeater', 'hare', 'breadbox', 'nest', 'rifle', 'skunk', 'barrel', 'keyboard', 'bun', 'hay', 'mosquito', 'coffin', 'newspaper', 'condom', 'deer', 'mouse', 'candle', 'cymbals', 'kumquat', 'mall', 'potato', 'candelabra', 'knot', 'lobster', 'crib', 'broccoli', 'fireworks', 'ant', 'crowbar', 'thermostat', 'caterpillar', 'papaya', 'ceiling', 'zucchini', 'pecan', 'wallet', 'radish', 'onion', 'crayon', 'pancake', 'canoe', 'casket', 'tunnel', 'scissors', 'cork', 'tofu', 'zebra', 'kangaroo', 'hamster', 'missile', 'dishwasher', 'bamboo', 'altar', 'otter', 'nacho', 'calculator', 'fence', 'rhubarb', 'stethoscope', 'library', 'tadpole', 'dollhouse', 'trashcan', 'guava', 'pomegranate', 'tamale', 'anteater', 'plum', 'oyster', 'cinnamon', 'bandage', 'elevator', 'mistletoe', 'marshmallow', 'custard', 'wristwatch', 'corkscrew', 'kazoo', 'kebab', 'granola', 'gargoyle', 'scone', 'mantis', 'parsnip', 'curtain', 'scorpion', 'crown', 'lemonade', 'wolf', 'bugle', 'graveyard', 'tumbleweed', 'plate', 'dragonfly', 'flag', 'crocodile', 'mushroom', 'beetle', 'cucumber', 'sloth', 'dough', 'sphinx', 'canyon', 'iguana', 'chalice', 'doormat', 'hairpin', 'aloe', 'scaffolding', 'platypus', 'brownie', 'casino', 'shrimp', 'grate', 'loudspeaker', 'tower', 'submarine', 'rug', 'ape', 'banner', 'syringe', 'hanger', 'cannon', 'kale', 'pothole', 'chili', 'waterspout', 'stadium', 'spacecraft', 'boar', 'cheese', 'celery', 'hammer', 'matchbook', 'coconut', 'beet', 'stegosaurus', 'chocolate', 'muffin', 'turnip', 'wire', 'chandelier', 'forklift', 'fondue', 'gazebo', 'wheelbarrow', 'melon', 'earpiece', 'paintbrush', 'bib', 'strongbox', 'steamroller', 'breadfruit', 'dishrag', 'anchor', 'fountain', 'parachute', 'sink', 'radiator', 'burner', 'llama', 'playground', 'okra', 'gramophone', 'burlap', 'earwig', 'calipers', 'spinach', 'couch', 'parsley', 'koala', 'bobsleigh', 'highlighter', 'silverfish', 'greenhouse', 'blimp', 'tray', 'mandolin', 'spareribs', 'gecko', 'volcano', 'cabbage', 'kettle', 'antenna', 'panda', 'microchip', 'toolbox', 'weasel', 'hairbrush', 'squeegee', 'flashbulb', 'pretzel', 'sawmill', 'joystick', 'persimmon', 'wand', 'graffiti', 'giraffe', 'chinchilla', 'sundial', 'beaker', 'honeycomb', 'fishnet', 'odometer', 'scallion', 'chess', 'photocopier', 'chair', 'locker', 'newsstand', 'reef', 'cheetah', 'salsa', 'auditorium', 'extinguisher', 'battery', 'dildo', 'chestnut', 'manhole', 'gooseberry', 'salamander', 'spearmint', 'hotplate', 'moth', 'lizard', 'beaver', 'nutmeg', 'cannabis', 'cogwheel', 'bolt', 'hut', 'robot', 'shield', 'chameleon', 'defibrillator', 'bison', 'geyser', 'raccoon', 'coleslaw', 'thermos', 'macadamia', 'compass', 'ruler', 'helicopter', 'manatee', 'pinwheel', 'cherry', 'pickle', 'projector', 'guillotine', 'tapioca', 'birdcage', 'coffeepot', 'sharpener', 'baguette', 'gearshift', 'pier', 'drain', 'artichoke', 'oscilloscope', 'cube', 'stapler', 'bubble', 'ukulele', 'tyrannosaurus', 'winterberry', 'cauldron', 'cassette', 'meteorite', 'urinal', 'hazelnut', 'rainbow', 'sieve', 'shredder', 'hovercraft']
    
    def __call__(self, probabilities):
        self.check_input(probabilities)
        sorted_indices = np.flip(np.argsort(probabilities), axis=-1)
        return np.take(self.categories, sorted_indices, axis=-1)

class miniecosetZejinProbabilitiesTo112ClassesMapping(DecisionMapping):
    """Return the WNIDs sorted by probabilities."""
    def __init__(self):
        self.categories =  ['man', 'woman', 'child', 'monkey', 'elephant', 'dog', 'pig', 'cat', 'tiger', 'leopard', 'zebra', 'wolf', 'camel', 'kangaroo', 'deer',
                      'rhino', 'cow', 'goat', 'mouse', 'squirrel', 'bird', 'snake', 'turtle', 'frog', 'spider', 'rabbit', 'alligator', 'cockroach', 'chicken',
                      'wasp', 'butterfly', 'snail', 'jellyfish', 'worm', 'grasshopper', 'lizard', 'mosquito', 'lobster', 'ant', 'beetle', 'seashell', 'fire',
                      'apple', 'flower', 'chestnut', 'egg', 'lemon', 'banana', 'fern', 'cactus', 'corn', 'lettuce', 'grape', 'rice', 'onion', 'oyster', 'aloe',
                      'parsley', 'bubble', 'cherry', 'phone', 'gun', 'book', 'ball', 'clock', 'bag', 'camera', 'cup', 'key', 'pizza', 'bell', 'toilet', 'knife',
                      'hamburger', 'axe', 'mailbox', 'lamp', 'bread', 'shovel', 'spoon', 'bed', 'table', 'ship', 'bus', 'bridge', 'computer', 'train', 'truck',
                      'refrigerator', 'airplane', 'lighthouse', 'piano', 'television', 'bicycle', 'windmill', 'barrel', 'canoe', 'cannon', 'hut', 'helicopter',
                      'keyboard','oven','chair', 'bear','boat','bottle',
                      'car', 'giraffe', 'fish', 'melon', 'papaya','pineapple']
    
    def __call__(self, probabilities):
        self.check_input(probabilities)
        sorted_indices = np.flip(np.argsort(probabilities), axis=-1)
        return np.take(self.categories, sorted_indices, axis=-1)

# Only 16 categories
class ImageNetProbabilitiesTo16ClassesMapping(DecisionMapping):
    """Return the 16 class categories sorted by probabilities"""

    def __init__(self, aggregation_function=None):
        if aggregation_function is None:
            aggregation_function = np.mean
        self.aggregation_function = aggregation_function
        self.categories = hc.get_human_object_recognition_categories()

    def __call__(self, probabilities):
        self.check_input(probabilities)

        aggregated_class_probabilities = []
        c = hc.HumanCategories()

        for category in self.categories:
            indices = c.get_imagenet_indices_for_category(category)
            values = np.take(probabilities, indices, axis=-1)
            aggregated_value = self.aggregation_function(values, axis=-1)
            aggregated_class_probabilities.append(aggregated_value)
        aggregated_class_probabilities = np.transpose(aggregated_class_probabilities)
        sorted_indices = np.flip(np.argsort(aggregated_class_probabilities, axis=-1), axis=-1)
        return np.take(self.categories, sorted_indices, axis=-1)

class ImageNet16ProbabilitiesTo16ClassesMapping(DecisionMapping):
    """Return the 16 class categories sorted by probabilities"""

    def __init__(self, aggregation_function=None):
        if aggregation_function is None:
            aggregation_function = np.mean
        self.aggregation_function = aggregation_function
        self.categories = hc.get_human_object_recognition_categories()

    def __call__(self, probabilities):
        self.check_input(probabilities)

        aggregated_class_probabilities = []
        c = hc.ImageNet16Shape2TexureHumanCategories()

        for category in self.categories:
            indices = c.get_imagenet_indices_for_category(category)
            values = np.take(probabilities, indices, axis=-1)
            aggregated_value = self.aggregation_function(values, axis=-1)
            aggregated_class_probabilities.append(aggregated_value)
        aggregated_class_probabilities = np.transpose(aggregated_class_probabilities)
        sorted_indices = np.flip(np.argsort(aggregated_class_probabilities, axis=-1), axis=-1)
        return np.take(self.categories, sorted_indices, axis=-1)

class ecosetTexture2ShapeProbabilitiesTo16ClassesMapping(DecisionMapping):
    """Return the 16 class categories sorted by probabilities"""

    def __init__(self, aggregation_function=None):
        if aggregation_function is None:
            aggregation_function = np.mean
        self.aggregation_function = aggregation_function
        self.categories = hc.get_human_object_recognition_categories()

    def __call__(self, probabilities):
        self.check_input(probabilities)

        aggregated_class_probabilities = []
        c = hc.ecosetShape2TexureHumanCategories()

        for category in self.categories:
            indices = c.get_imagenet_indices_for_category(category)
            values = np.take(probabilities, indices, axis=-1)
            aggregated_value = self.aggregation_function(values, axis=-1)
            aggregated_class_probabilities.append(aggregated_value)
        aggregated_class_probabilities = np.transpose(aggregated_class_probabilities)
        sorted_indices = np.flip(np.argsort(aggregated_class_probabilities, axis=-1), axis=-1)

        return np.take(self.categories, sorted_indices, axis=-1)

class MiniEcosetTexture2ShapeProbabilitiesTo16ClassesMapping(DecisionMapping):
    """Return the 16 class categories sorted by probabilities"""

    def __init__(self, aggregation_function=None):
        if aggregation_function is None:
            aggregation_function = np.mean
        self.aggregation_function = aggregation_function
        self.categories = hc.get_human_object_recognition_categories()

    def __call__(self, probabilities):
        self.check_input(probabilities)

        aggregated_class_probabilities = []
        c = hc.MiniecosetShape2TexureHumanCategories()

        for category in self.categories:
            indices = c.get_imagenet_indices_for_category(category)
            # print(f"debug indices:{indices} & probabilities:{probabilities}")
            values = np.take(probabilities, indices, axis=-1)
            aggregated_value = self.aggregation_function(values, axis=-1)
            aggregated_class_probabilities.append(aggregated_value)
        aggregated_class_probabilities = np.transpose(aggregated_class_probabilities)
        sorted_indices = np.flip(np.argsort(aggregated_class_probabilities, axis=-1), axis=-1)
        # print(f"debug sorted_indices:{sorted_indices}")
        return np.take(self.categories, sorted_indices, axis=-1)