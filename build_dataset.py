from argparse import ArgumentParser
from glob import glob
from json import dump
from math import gcd
from os import mkdir
from os.path import basename, exists
from random import randint, random, seed, shuffle, uniform
from shutil import copy, rmtree

import cv2
import numpy as np
from impl.aruco import get_marker
from impl.effects import rotate3d
from impl.shadows import lines, perlin
from tqdm import tqdm

from impl.utils import ordered_corners

def get_rotation(original, ordered):

    ori = [(original[i], original[(i + 1) % 8]) for i in range(0, 8, 2)]
    ord = [(ordered[i], ordered[(i + 1) % 8]) for i in range(0, 8, 2)]

    sides_ori = [sorted((ori[i], ori[(i + 1) % 4])) for i in range(4)]
    sides_ord = [sorted((ord[i], ord[(i + 1) % 4])) for i in range(4)]
    
    rot = sides_ord.index(sides_ori[0])
   
    return rot

normal = lambda gen, loc, scale: gen.normal(loc, scale)

if __name__ == '__main__':

    parser = ArgumentParser(description = 'FlyingArUco v2 dataset builder.')
    parser.add_argument('source_dir', help = 'where to find source images')
    parser.add_argument('output_dir', help = 'where to store generated images + labels')
    parser.add_argument('-l', '--luma', help = 'use background luma in overlaid markers', action='store_true')
    parser.add_argument('-b', '--borders', help = 'use variable width for marker borders', action='store_true')
    parser.add_argument('-nb', '--no_borders', help = 'set border width to zero', action = 'store_true')
    parser.add_argument('-r', '--reflections', help = 'allow permeability in dark areas to simulate ink reflections', action = 'store_true')
    #parser.add_argument('--blur', help = 'add per-image blur', action = 'store_true')
    args = parser.parse_args()

    # Initialize random seed for replicability

    seed(0)
    np.random.seed(0)

    # Initialize random generators
    
    gen_x = np.random.default_rng(0)
    gen_y = np.random.default_rng(0)

    # Paths

    source_dir = args.source_dir
    output_dir = args.output_dir

    # Image dims

    t_width = 640
    t_height = 360

    # Marker filling

    max_markers = 20
    retries = 5

    if exists(output_dir):
        rmtree(output_dir)
    mkdir(output_dir)

    # Create images

    for path in tqdm(glob(source_dir + '/*.jpg')): 

        pic = cv2.imread(path)
        
        if pic.shape[0] > pic.shape[1]:
            pic = cv2.rotate(pic, cv2.ROTATE_90_CLOCKWISE)

        if pic.shape[1] >= t_width and pic.shape[0] >= t_height:
            centerw, centerh = pic.shape[1]//2, pic.shape[0]//2
            pic = pic[centerh - t_height//2:centerh + t_height//2, centerw - t_width//2:centerw + t_width//2]
    
            height, width = pic.shape[:2]

            markers = np.zeros((height, width, 4), dtype=np.uint8)
            occupied = np.zeros((height, width), dtype=np.uint8)

            #min_size = 2 * gcd(width, height)
            min_size = 32

            all_corners = []
            ids = []
            rots = []

            for i in range(max_markers):
                for r in range(retries):

                    size = randint(min_size, min(width, height))
                    m_size = size # Marker size (for later downsampling)

                    x, y = randint(0, width - size), randint(0, height - size)

                    b_width = random()
                    fill_color = [255,255,255]

                    if args.borders == False:
                        b_width = 1.0
                    
                    if args.no_borders == True:
                        b_width = 0
                    
                    if b_width == 0:
                        fill_color = [0,0,0]

                    if random() < 0.5: # Real markers
                        real = True
                        id = randint(0, 249)
                        marker, corners = get_marker(id, size = m_size, border_width = b_width)

                    else: # Fake markers
                        real = False
                        type_val = random()

                        if type_val < 0.8: # Designs
                            marker, corners = get_marker(251, size = m_size, border_width = b_width)

                            content = np.ones(marker.shape[:2])
                            for f in [lines, perlin]:
                                if random() > 0.5: 
                                    content *= cv2.resize(f(64, 64), (marker.shape[1], marker.shape[0]))

                            mask = get_marker(250, size = m_size, border_width = 0)[0][:,:,3] / 255.0
                            content = (1 - mask) + content * mask
                            marker[:,:,:3] = np.clip(np.multiply(marker[:,:,:3], np.expand_dims(content, -1)), 0, 255)

                        elif type_val < 0.9: # Full black
                            if random() > 0.5:
                                b_width = 0
                                fill_color = [0,0,0]
                            marker, corners = get_marker(250, size = m_size, border_width = b_width)

                        else:
                            id = randint(0, 249) # Inverted
                            marker, corners = get_marker(id, size = m_size, border_width = 0.0)
                            mask = marker[:,:,3]
                            mask = np.repeat(mask[..., np.newaxis], 3, axis = 2) / 255.0
                            marker[:,:,:3] = (1 - mask) * marker[:,:,:3] + mask * (255 - marker[:,:,:3])

                        if random() < 0.2: # Color shift
                            real = False
                            change = [uniform(0, 0.33), uniform(0.33, 0.66), uniform(0.66, 1.0)]
                            shuffle(change)
                            for c in range(3): 
                                marker[:,:,c] = marker[:,:,c] * change[c]
                                fill_color[c] = fill_color[c] * change[c]

                    full_border, _ = get_marker(0, size = m_size)

                    extreme = 75
                    scale = extreme / 3.5

                    rotx, roty, rotz = normal(gen_x, 0, scale), normal(gen_y, 0, scale), uniform(0, 360)

                    f_mult = uniform(1.0, 4.0)
                    
                    marker, transform = rotate3d(marker, rotx, roty, rotz, f_mult, fill_color=fill_color)
                    marker = cv2.resize(marker, (size, size), interpolation=cv2.INTER_AREA)

                    full_border, _ = rotate3d(full_border, rotx, roty, rotz, f_mult)
                    full_border = cv2.resize(full_border, (size, size), interpolation=cv2.INTER_AREA)

                    # Counterclockwise sort: https://pavcreations.com/clockwise-and-counterclockwise-sorting-of-coordinates/

                    corners = ordered_corners([c[0] for c in corners], [c[1] for c in corners])
                    corners = [[corners[i], corners[i + 1]] for i in range(0, 8, 2)]

                    corners = [[c[0] + m_size/2, c[1] + m_size/2] for c in corners]
                    corners = np.array(corners, dtype = np.float32).reshape(-1,1,2).astype(np.float32)

                    corners = [c[0] for c in cv2.perspectiveTransform(corners, transform).tolist()]
                    corners = [[v / (2 * m_size) * size for v in c] for c in corners]

                    # Get rotation

                    unpacked = [v for c in corners for v in c]
                    ordered = ordered_corners([corners[i][0] for i in range(4)], [corners[i][1] for i in range(4)])
                    rot = get_rotation(unpacked, ordered)

                    if (occupied[y:y + size, x:x + size] != 255).all():

                        markers[y:y + size, x:x + size] = marker
                        occupied[y:y + size, x:x + size] = full_border[:,:,3]

                        all_corners.append(corners)
                        ids.append(id)
                        rots.append(rot)

                        break

            if len(all_corners) > 0:

                # Overlay markers

                if args.luma:
                    luma = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                    luma = np.expand_dims(luma, -1)
                    luma = np.repeat(luma, 3, axis = 2)
                    markers[:,:,:3] = np.clip(markers[:,:,:3] * luma, 0, 255)

                if args.reflections:
                    mask = markers[:,:,3] / 255.0
                    mask = np.expand_dims(mask, -1)
                    mask = np.repeat(mask, 3, axis = 2)
                    pic = pic * (1 - mask) + markers[:,:,:3] * mask

                else:
                    mask = markers[:,:,3] / 255.0
                    mask = np.expand_dims(mask, -1)
                    mask = np.repeat(mask, 3, axis = 2)
                    pic = pic * (1 - mask) + markers[:,:,:3] * mask

                # Save image

                cv2.imwrite(f'{output_dir}/{basename(path)}', pic)

                # Save labels

                with open(f'{output_dir}/{basename(path)[:-4]}.txt', 'w') as f:

                    for corners, id, rot in zip(all_corners, ids, rots):

                        # YOLO format: <class> <x_center> <y_center> <width> <height>

                        x_coords = [c[0] for c in corners]
                        y_coords = [c[1] for c in corners]

                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)

                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        width = x_max - x_min
                        height = y_max - y_min

                        f.write(f'{id} {x_center} {y_center} {width} {height}\n')

    # Copy brightness.csv if it exists
    brightness_file = f'{source_dir}/brightness.csv'
    if exists(brightness_file):
        copy(brightness_file, f'{output_dir}/brightness.csv')