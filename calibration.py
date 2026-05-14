import numpy as np

def flip_flex_values(flex_array):
    #NAPRAW!!!!!
    flipped = 250 - np.array(flex_array, dtype=float)
    return flipped.astype(float).tolist()


def align_to_dataset(parsed):
    f1, f2, f3, f4, f5 = parsed["FLEX"]
    gyr_x, gyr_y, gyr_z = parsed["GYR"]
    accbx, accby, accbz = parsed["ACC_body"]

    return [
        f1, f2, f3, f4, f5,
        gyr_x, gyr_y, gyr_z,
        accbx, accby, accbz
    ]

