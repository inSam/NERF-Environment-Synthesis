from dataclasses import dataclass, asdict

import numpy as np
import cv2
import json

@dataclass
class PBRT:
    x_res: int
    y_res: int
    filename: str
    fov: float
    

@dataclass
class CameraInfo:
    camera_angle_x: float
    camera_angle_y: float
    f1_x: float
    f1_y: float
    w: int = 1920
    h: int = 1080
    aabb_scale: int = 4
    k1: float = 0
    k2: float = 0
    p1: float = 0
    p2: float = 0

def bistro_cafe_build(x, y, z, u, v, w, outfile):
    """Construct pbrt file given a certain camera position and target in the reference coordinate system 
        1. Outputs the pbrt file
        2. Constructs the camera info class

    Args:
        x (float): x pos of eye
        y (float): y pos of eye
        z (float): z pos of eye
        u (float): x pos of target position
        v (float): y pos of target position
        w (float): z pos of target position
        outfile (str): name of the output file
    """
    height = 0
    width = 0
    fov = 0
    
    

def main():
    PBRT_PATH = ""
    OUT_PATH = "./transform.json"
    x = CameraInfo(45., 45., 10., 10.)
        
    out_json = asdict(x)
    with open(OUT_PATH, "w") as outfile:
        json.dump(out_json, outfile, indent=2)

if __name__ == '__main__':
    main()