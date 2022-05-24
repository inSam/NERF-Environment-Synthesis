import re
import math
import tempfile
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

def get_camera_info(infile: str) -> CameraInfo:
    """Generates the camera information from the pbrt tempalte file

    Args:
        infile (str): Path to the pbrt template file

    Returns:
        CameraInfo: Data class containing camera information
    """
    height = 0
    width = 0
    fov = 0 # In degrees
    with open(infile, 'r') as template_f:
        for line in template_f:
            if "xresolution" in line:
                m = re.search(r'\[\s?(\d+)\s?\]', line)
                if m != None:
                    width = int(m.group(1))
            if "yresolution" in line:
                m = re.search(r'\[\s?(\d+)\s?\]', line)
                if m != None:
                    height = int(m.group(1))
            if "float fov" in line:
                m = re.search(r'\[\s?(\d+)\s?\]', line)
                if m != None:
                    fov = float(m.group(1))
    
    assert(height != 0 and width != 0)
    assert(fov != 0)
    
    # convert to radians     
    fov = math.radians(fov)
    fl = min(height, width) / (2 * math.tan(fov/2))
    fov_greater = 2 * math.atan(max(height, width) / (fl * 2)) * 2
    if width > height:
        return CameraInfo(fov_greater, fov, fl, fl, width, height)
    else:
        return CameraInfo(fov, fov_greater, fl, fl, width, height)
    
def render_pbrt(x, y, z, u, v, w, infile, outimg) -> None:
    """Render pbrt file given a certain camera position and target in the reference coordinate system 
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
    eye_pos = " ".join(str(i) for i in [x, y, z])
    look_pos = " ".join(str(i) for i in [u, v, w])

    with tempfile.TemporaryFile('w') as tmp_fp:
        with open(infile, 'r') as template_f:
            # Generate pbrt file with correct camera params
            for line in template_f:
                line = line.replace("[CAMERA_POS]", eye_pos)
                line = line.replace("[CAMERA_TGT]", look_pos)
                line = line.replace("[OUTFILE]", '"' + outimg + '"')
                tmp_fp.write(line)
            # Render the pbrt file


def pathgen_bistro()-> tuple[float, float, float, float, float, float]: 
    eye_x, eye_y, eye_z = -11.7, 2.5, 6.5
    tgt_x, tgt_y, tgt_z = -8.5, 2.8, 8
    
    
    return (1,1,1,1,1,1)

def main():
    PBRT_PATH = ""
    OUT_PATH = "./transform.json"

    cam_info = get_camera_info("bistro_cafe_template.pbrt")

    #build_pbrt(0,0,0,0,0,0, "bistro_cafe_template.pbrt")
        
    out_json = asdict(cam_info)
    with open(OUT_PATH, "w") as outfile:
        json.dump(out_json, outfile, indent=2)

if __name__ == '__main__':
    main()