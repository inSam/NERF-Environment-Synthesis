import re
import math
import tempfile
from dataclasses import dataclass, asdict, field


import numpy as np
import cv2
import json


def normalize_vector(vector):
    return vector / (np.linalg.norm(vector))


def get_lookat_matrix(position_vector, front_vector, up_vector) -> np.ndarray:
    m1 = np.zeros([4, 4], dtype=np.float32)
    m2 = np.zeros([4, 4], dtype=np.float32)

    z = normalize_vector(-front_vector)
    x = normalize_vector(np.cross(up_vector, z))
    y = np.cross(z, x)

    m1[0, :3] = x
    m1[1, :3] = y
    m1[2, :3] = z
    m1[3, 3] = 1.0

    m2[0, 0] = m2[1, 1] = m2[2, 2] = 1.0
    m2[3, :3] = -position_vector
    m2[3, 3] = 1.0

    return np.matmul(m1, m2)


@dataclass
class PBRT:
    x_res: int
    y_res: int
    filename: str
    fov: float


@dataclass
class CameraInfo:
    """Data class containing information about the camera
    Independent of individual frames

    The format is currently using the OpenCV model for a pinhole camera
    Currently assumes that all input images are rendered with a distortion free perspective camera
    """
    camera_angle_x: float
    camera_angle_y: float
    f1_x: float
    f1_y: float
    w: int
    h: int
    up_vec: np.ndarray
    aabb_scale: int = 4
    k1: float = 0
    k2: float = 0
    p1: float = 0
    p2: float = 0
    cx: float = field(init=False)
    cy: float = field(init=False)

    def __post_init__(self):
        self.cx = self.w / 2
        self.cy = self.h / 2


def get_camera_info(infile: str) -> CameraInfo:
    """Generates the camera information from the pbrt tempalte file

    Args:
        infile (str): Path to the pbrt template file

    Returns:
        CameraInfo: Data class containing camera information
    """
    height = 0
    width = 0
    up_vec = np.array([])
    # In degrees
    # The given FOV is angle of the fructum for the smaller of the two image axes
    fov = 0
    with open(infile, 'r') as template_f:
        for line in template_f:
            if "xresolution" in line:
                m = re.search(r'\[\s?(\d+)\s?\]', line)
                if m is not None:
                    width = int(m.group(1))
            if "yresolution" in line:
                m = re.search(r'\[\s?(\d+)\s?\]', line)
                if m is not None:
                    height = int(m.group(1))
            if "float fov" in line:
                m = re.search(r'\[\s?(\d+)\s?\]', line)
                if m is not None:
                    fov = float(m.group(1))
            if "LookAt" in line:
                up_vec = np.array(line.split(' ')[-3:])

    assert(height != 0 and width != 0)
    assert(fov != 0)

    # convert to radians
    fov = math.radians(fov)

    # Calculate the FOV for both the X and Y direction
    fl = min(height, width) / (2 * math.tan(fov / 2))
    fov_greater = 2 * math.atan(max(height, width) / (fl * 2)) * 2
    if width > height:
        return CameraInfo(fov_greater, fov, fl, fl, width, height, up_vec)
    else:
        return CameraInfo(fov, fov_greater, fl, fl, width, height, up_vec)


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


def pathgen_bistro() -> tuple[float, float, float, float, float, float]:
    eye_x, eye_y, eye_z = -11.7, 2.5, 6.5
    tgt_x, tgt_y, tgt_z = -8.5, 2.8, 8

    return (eye_x, eye_y, eye_z, tgt_x, tgt_y, tgt_z)


def main():
    PBRT_PATH = ""
    OUT_PATH = "./transform.json"

    cam_info = get_camera_info("bistro_cafe_template.pbrt")
    eye_x, eye_y, eye_z, tgt_x, tgt_y, tgt_z = pathgen_bistro()
    position_vector = np.array([eye_x, eye_y, eye_z])
    front_vector = np.array([tgt_x, tgt_y, tgt_z])
    world_to_camera = get_lookat_matrix(position_vector, front_vector, cam_info.up_vec)
    print(world_to_camera)

    # build_pbrt(0,0,0,0,0,0, "bistro_cafe_template.pbrt")

    out_json = asdict(cam_info)
    with open(OUT_PATH, "w") as outfile:
        json.dump(out_json, outfile, indent=2)


if __name__ == '__main__':
    main()
