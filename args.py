import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a dataset of PBRT renderings for Nerf")

    parser.add_argument("--template", type=str, default="/home/samx/pbrt-v4-scenes/bistro/bistro_cafe_template.pbrt",
                        help="input path to the pbrt template")
    parser.add_argument("--aabb_scale", default=4, choices=["1", "2", "4", "8", "16"],
                        help="large scene scale factor. 1 scene fits in unit cube; power of 2 up to 16")
    parser.add_argument("--num_samples", type=int, default=280,
                        help="Numbers of sample to generate")
    parser.add_argument("--db", type=str, default="bistro",
                        choices=["bistro", "crown"], help="Decide which pbrt template to render")
    parser.add_argument("--out", default="./", help="output path")
    args = parser.parse_args()
    return args
