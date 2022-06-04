import os
import re
import math
from dataclasses import dataclass, asdict, field
from args import parse_args

import numpy as np
import cv2
import json

template_keys = {
    "bistro": "/home/samx/pbrt-v4-scenes/bistro/bistro_cafe_template.pbrt",
    "crown": "/home/samx/pbrt-v4-scenes/crown/crown_template.pbrt"
}


def generate_transform_matrix(pos, rot):
    def Rx(theta):
        return np.matrix([[1, 0, 0],
                          [0, np.cos(theta), -np.sin(theta)],
                          [0, np.sin(theta), np.cos(theta)]])

    def Ry(theta):
        return np.matrix([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])

    def Rz(theta):
        return np.matrix([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])

    R = Rz(rot[2]) * Ry(rot[1]) * Rx(rot[0])
    xf_rot = np.eye(4)
    xf_rot[:3, :3] = R

    xf_pos = np.eye(4)
    xf_pos[:3, 3] = pos  # - average_position

    # barbershop_mirros_hd_dense:
    # - camera plane is y+z plane, meaning: constant x-values
    # - cameras look to +x

    # Don't ask me...
    extra_xf = np.matrix([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]])
    # NerF will cycle forward, so lets cycle backward.
    shift_coords = np.matrix([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]])
    xf = shift_coords @ extra_xf @ xf_pos
    assert np.abs(np.linalg.det(xf) - 1.0) < 1e-4
    xf = xf @ xf_rot
    return xf


def sharpness(imagePath):
    image = cv2.imread(imagePath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def normalize_vector(vector):
    return vector / (np.linalg.norm(vector))


def get_lookat_matrix(eye_vector, center_vector, up_vector) -> np.ndarray:
    m1 = np.zeros([4, 4], dtype=np.float32)
    m2 = np.identity(4, dtype=np.float32)

    z = normalize_vector(eye_vector - center_vector)
    x = normalize_vector(np.cross(up_vector, z))
    y = np.cross(z, x)

    m1[0, :3] = x
    m1[1, :3] = y
    m1[2, :3] = z
    m1[3, 3] = 1.0

    m2[0, 0] = m2[1, 1] = m2[2, 2] = 1.0
    m2[:3, 3] = -eye_vector
    m2[3, 3] = 1.0

    return np.matmul(m1, m2)


@dataclass
class PBRT_Frame:
    eye_pos: np.ndarray
    center_pos: np.ndarray
    filename: str


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
                up_vec = np.array([float(i.strip())
                                  for i in line.split(' ')[-3:]])

    assert(height != 0 and width != 0)
    assert(fov != 0)

    # convert to radians
    fov = math.radians(fov)

    # Calculate the FOV for both the X and Y direction
    fl = min(height, width) / (2 * math.tan(fov / 2))
    fov_greater = 2 * math.atan(max(height, width) / (fl * 2))
    if width > height:
        return CameraInfo(fov_greater, fov, fl, fl, width, height, up_vec)
    else:
        return CameraInfo(fov, fov_greater, fl, fl, width, height, up_vec)


def render_pbrt(eye_pos, center_pos, infile, outimg) -> None:
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
    eye = " ".join(str(i) for i in eye_pos)
    look = " ".join(str(i) for i in center_pos)
    TEMPLATE_DIR = os.path.dirname(infile)
    tmp_file = os.path.join(TEMPLATE_DIR, "temp.pbrt")

    with open(tmp_file, 'w') as tmp_fp:
        with open(infile, 'r') as template_f:
            # Generate pbrt file with correct camera params
            for line in template_f:
                line = line.replace("[CAMERA_POS]", eye)
                line = line.replace("[CAMERA_TGT]", look)
                line = line.replace("[OUTFILE]", '"' + outimg + '"')
                tmp_fp.write(line)

    # Render the pbrt file
    os.system(f"pbrt --gpu --display-server localhost:14158 {tmp_file}")
    os.system(f"imgtool denoise-optix {outimg} --outfile {outimg}")


def pathgen_bistro(outfile: str) -> PBRT_Frame:
    eye_x, eye_y, eye_z = [-11.7 + np.random.normal(-1, 1),
                           2.5 + np.random.normal(-1, 1),
                           6.5 + np.random.normal(-1, 1)]
    # eye_x, eye_y, eye_z = [-11.7,
    #                        2.5,
    #                        6.5]

    # Randomly generating a direction to look at for the camera
    # A random normal distribution of coordinates gives you a uniform distribution of directions.
    look_x, look_y, look_z = normalize_vector(np.random.randn(3))

    tgt_x, tgt_y, tgt_z = eye_x + look_x, eye_y + look_y, eye_z + look_z

    return PBRT_Frame(np.array([eye_x, eye_y, eye_z]),
                      np.array([tgt_x, tgt_y, tgt_z]),
                      outfile)


def pathgen_crown(outfile: str) -> PBRT_Frame:
    eye_to_dist_scale = 34.4419802

    tgt_x, tgt_y, tgt_z = [0, 11, -10]
    eye = np.random.normal(size=3)
    eye[1] = max(eye[1], -0.01)
    eye_x, eye_y, eye_z = normalize_vector(
        eye) * eye_to_dist_scale + [tgt_x, tgt_y, tgt_z]
    #eye_x, eye_y, eye_z = [0, 5.5, 24]

    return PBRT_Frame(np.array([eye_x, eye_y, eye_z]),
                      np.array([tgt_x, tgt_y, tgt_z]),
                      outfile)


def main():
    args = parse_args()
    TEMPLATE_PATH = template_keys[args.db]

    # Construct the Output directory
    OUT_DIR = os.path.join(args.out, "images", args.db)
    # Check to make sure that the output directory eexists
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    assert(os.path.isfile(TEMPLATE_PATH))
    assert(os.path.isdir(OUT_DIR))

    cam_info = get_camera_info(TEMPLATE_PATH)

    # print(cam_info.up_vec)
    # print(R)
    #print(R @ np.array([0,1,0,1]))
    # return

    frames = []
    up = np.zeros(3)
    for i in range(args.num_samples):
        out = os.path.join(OUT_DIR, f"{i:0>3d}.exr")

        frame = None
        if args.db == "bistro":
            frame = pathgen_bistro(out)
        elif args.db == "crown":
            frame = pathgen_crown(out)
        else:
            raise Exception("Template not supported")

        render_pbrt(frame.eye_pos, frame.center_pos,
                    TEMPLATE_PATH, frame.filename)

        #frame.eye_pos = (R @ np.append(frame.eye_pos, 1))[:3]
        #frame.center_pos = (R @ np.append(frame.center_pos, 1))[:3]

        # frame.eye_pos = frame.eye_pos[[0, 2, 1]]
        # frame.center_pos = frame.center_pos[[0, 2, 1]]
        w2c = get_lookat_matrix(
            np.array(frame.eye_pos), np.array(frame.center_pos), np.array([0, 1, 0]))
        c2w = np.linalg.inv(w2c)
        #c2w = R @ c2w
        # c2w[0:3,2] *= -1 # flip the y and z axis
        # c2w[0:3,1] *= -1
        # c2w = c2w[[1,0,2,3],:] # swap y and z
        # c2w[2,:] *= -1 # flip whole world upside down
        up += c2w[0:3,1]
        # sharp = sharpness(frame.filename)
        frames.append({
            "file_path": f"./{i:0>3d}.exr",
            # "sharpness": sharp,
            "transform_matrix": c2w
            # "transform_matrix": generate_transform_matrix([0,0,1], [0, 0, 0]).tolist()
            # "transform_matrix": np.eye(4).tolist()
        })

    up = up / np.linalg.norm(up)
    print("up vector was", up)
    # Create Rotation matrix to adjust up vector to [0, 0, 1]
    R = rotmat([0,1,0], [0, 0, 1])
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1
    avglen = 0.
    for f in frames:
        f["transform_matrix"] = np.matmul(R, f["transform_matrix"])
        avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
    avglen /= len(frames)
    for f in frames:
        f["transform_matrix"][0:3, 3] *= 2.0 / avglen
        f["transform_matrix"] = f["transform_matrix"].tolist()

    out_json = asdict(cam_info)
    del out_json["up_vec"]
    out_json["frames"] = frames
    with open(os.path.join(OUT_DIR, args.db + ".json"), "w") as outfile:
        json.dump(out_json, outfile, indent=2)


if __name__ == '__main__':
    main()
