import numpy as np
import os
import Augmentor

# p = Augmentor.Pipeline('data/imgs_192')
# p.ground_truth('data/masks_192')
def Augment_imgs():
    p = Augmentor.Pipeline('data/imgs')
    p.ground_truth('data/masks')

    # p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)        # 插值鬼影
    p.rotate_random_90(probability=1)
    # p.skew_corner(probability= 0.7, magnitude=0.3)                                  # 插值鬼影
    p.zoom(probability=1, min_factor=1.0, max_factor= 1.5)
    # p.crop_centre(probability=1,percentage_area=0.66666)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    # p.random_brightness(probability=1, min_factor=0.7, max_factor=1.2)
    # p.random_contrast(probability=1, min_factor=0.7, max_factor=1.2)
    # p.random_color(probability=1, min_factor=0.5, max_factor=1.5)

    p.sample(7130)
    # p.process()



def Augment_imgslarge():
    p = Augmentor.Pipeline('data/imgslarge')
    p.ground_truth('data/maskslarge')

    p.rotate_random_90(probability=1)
    p.zoom(probability=1, min_factor=0.7, max_factor= 1.0)
    p.crop_centre(probability=1,percentage_area=0.66666)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)


    p.sample(6600)
    # p.process()

if __name__ == "__main__":
    Augment_imgslarge()
    