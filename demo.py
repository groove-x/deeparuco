from argparse import ArgumentParser

import cv2
import numpy as np
import tensorflow as tf
import torch
from impl.aruco import find_id
from impl.heatmaps import pos_from_heatmap
from impl.losses import weighted_loss
from impl.utils import marker_from_corners, ordered_corners
from tensorflow.keras.models import load_model
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
import time
from contextlib import contextmanager
from collections import defaultdict

class MeasureExecutionTime:
    def __init__(self, ignore_first=True):
        self.times = defaultdict(list)
        self.current_section = None
        self.ignore_first = ignore_first

    @contextmanager
    def __call__(self, section_name):
        self.current_section = section_name
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.times[section_name].append(elapsed)
            self.current_section = None

    def print_all(self, title="Execution Times"):
        print(f"\n{title}")
        print("-" * 50)
        max_name_length = max(len(name) for name in self.times.keys())
        for section, times in self.times.items():
            if self.ignore_first:
                times = times[1:]
            avg_time = sum(times) / len(times)
            print(f"  {section:<{max_name_length}} {avg_time*1000:>8.1f} msec")
        # print("-" * 50)
        # total_time = sum(sum(times) for times in self.times.values())
        # print(f"  {'total':<{max_name_length}} {total_time*1000:>8.1f} msec")

    def reset(self):
        self.times.clear()

    def get_total_time(self):
        return sum(sum(times) for times in self.times.values())


deeparuco_timer = MeasureExecutionTime()
opencv_timer = MeasureExecutionTime()


norm = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)

def process_markers(pic, detector, exp, args, refine_corners, decode_markers):
    # Prepare image for YOLOX
    preproc = ValTransform(legacy=False)
    img, _ = preproc(pic, None, exp.test_size)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()
    if args.device == "gpu":
        img = img.cuda()
        if args.fp16:
            img = img.half()

    # Detect markers
    with deeparuco_timer("detection"):
        with torch.no_grad():
            outputs = detector(img)
            outputs = postprocess(
                outputs, exp.num_classes, exp.test_conf,
                exp.nmsthre, class_agnostic=True
            )

    # Get detections
    if outputs[0] is None:
        print("No markers detected")
        return None

    detections = outputs[0].cpu()
    ratio = min(exp.test_size[0] / pic.shape[0], exp.test_size[1] / pic.shape[1])
    bboxes = detections[:, 0:4]
    bboxes /= ratio  # Scale back to original image size

    # Expanded bboxes
    xyxy = [
        [
            int(max(det[0] - (0.2 * (det[2] - det[0]) + 0.5), 0)),
            int(max(det[1] - (0.2 * (det[3] - det[1]) + 0.5), 0)),
            int(min(det[2] + (0.2 * (det[2] - det[0]) + 0.5), pic.shape[1] - 1)),
            int(min(det[3] + (0.2 * (det[3] - det[1]) + 0.5), pic.shape[0] - 1)),
        ]
        for det in bboxes
    ]

    # Crop and normalize
    crops_ori = [
        cv2.resize(pic[det[1] : det[3], det[0] : det[2]], (64, 64)) for det in xyxy
    ]

    # Output crops
    if args.get_crops:
        for i in range(len(crops_ori)):
            cv2.imwrite(f"crop_{i}.png", crops_ori[i])

    # Normalize (if not baseline!)
    if args.regressor != "reg_baseline":
        crops = [norm(crop) for crop in crops_ori]
    else:
        crops = crops_ori.copy()

    # Refine corners
    with deeparuco_timer("regression"):
        corners = refine_corners(np.array(crops)).numpy()

    # Convert to (x, y) pairs
    if args.regressor.split("_")[1] == "hmap":
        # Output hmaps
        if args.get_heatmaps:
            for i in range(corners.shape[0]):
                cv2.imwrite(f"map_{i}.png", norm(corners[i]) * 255)

        # Instantiate keypoint detector
        area = 75  # <- Expected area of the blobs to detect
        kp_params = cv2.SimpleBlobDetector_Params()
        if area > 0:
            kp_params.filterByArea = True
            kp_params.minArea = area * 0.8
            kp_params.maxArea = area * 1.2
        kp_detector = cv2.SimpleBlobDetector_create(kp_params)

        corners = [
            [(x, y) for x, y in zip(*pos_from_heatmap(pred, kp_detector))]
            for pred in corners
        ]

        # Discard detections if less than 4 corners
        keep = [len(cs) == 4 for cs in corners]
        xyxy, crops_ori, corners = zip(
            *[
                (det, crop, cs)
                for det, crop, cs, k in zip(xyxy, crops_ori, corners, keep)
                if k == True
            ]
        )

    else:
        corners = [[(pred[i], pred[i + 1]) for i in range(0, 8, 2)] for pred in corners]

    # Ensure corners are ordered
    corners = [
        ordered_corners([c[0] for c in cs], [c[1] for c in cs]) for cs in corners
    ]

    # Extract markers from corners (if 4 corners available)
    markers = []
    for crop, cs in zip(crops_ori, corners):
        marker = marker_from_corners(crop, cs, 32)
        # Grayscale and normalize
        markers.append(norm(cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)))

    if args.get_markers:
        for i in range(len(markers)):
            cv2.imwrite(f"marker_{i}.png", markers[i] * 255.0)

    # Get ids from markers
    with deeparuco_timer("decode"):
        decoder_out = np.round(decode_markers(np.array(markers)).numpy())
        ids, dists = zip(*[find_id(out) for out in decoder_out])

    # Visualize
    line_width = 2  # Line width for drawing detections
    for cs, det, id, dist in zip(corners, xyxy, ids, dists):
        # Pack 2-by-2
        cs = [(cs[i], cs[i + 1]) for i in range(0, 8, 2)]

        color = (0, 255, 0)
        if dist >= args.threshold:
            color = (0, 0, 255)

        width = det[2] - det[0]
        height = det[3] - det[1]

        for i in range(0, 4):
            p1 = (int(det[0] + cs[i][0] * width), int(det[1] + cs[i][1] * height))
            p2 = (
                int(det[0] + cs[(i + 1) % 4][0] * width),
                int(det[1] + cs[(i + 1) % 4][1] * height),
            )
            pic = cv2.line(pic, p1, p2, color, line_width, cv2.LINE_AA)

        pic = cv2.putText(
            pic,
            str(id),
            (det[0] + int(width / 2) - 20, det[1] + int(height / 2) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            line_width,
            cv2.LINE_AA,
        )

    return pic

if __name__ == "__main__":
    parser = ArgumentParser(description="DeepArUco v2 demo tool.")
    parser.add_argument("pic_path", help="input image path")
    parser.add_argument("out_path", help="path to store output")
    parser.add_argument(
        "-d", "--detector", help="marker detector to use", default="det_luma_bc_s"
    )
    parser.add_argument(
        "-r", "--regressor", help="corner refinement model to use", default="reg_hmap_8"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        help="threshold to filter after decoding step",
        type=int,
        default=9,
    )
    parser.add_argument(
        "-hm",
        "--get_heatmaps",
        help="also return heatmaps (if applicable)",
        action="store_true",
    )
    parser.add_argument(
        "-c", "--get_crops", help="also return cropped detections", action="store_true"
    )
    parser.add_argument(
        "-m", "--get_markers", help="also return rectified markers", action="store_true"
    )
    # YOLOX specific arguments
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run model on (cpu/gpu)",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    args = parser.parse_args()

    # Paths
    model_dir = "./models"

    # Load models
    exp = get_exp(args.exp_file, args.name)
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    detector = exp.get_model()
    if args.device == "gpu":
        detector.cuda()
        if args.fp16:
            detector.half()
    detector.eval()

    # Load checkpoint
    ckpt_file = args.ckpt
    ckpt = torch.load(ckpt_file, map_location="cpu")
    detector.load_state_dict(ckpt["model"])

    regressor = load_model(
        f"{model_dir}/{args.regressor}.h5",
        custom_objects={"weighted_loss": weighted_loss},
    )
    decoder = load_model(f"{model_dir}/dec_new.h5")

    # Use graph execution for tf models
    @tf.function(reduce_retracing=True)
    def refine_corners(crops):
        return regressor(crops)

    @tf.function(reduce_retracing=True)
    def decode_markers(markers):
        return decoder(markers)

    # Load image
    pic = cv2.imread(args.pic_path)

    # Run DeepAruco++ and measure time
    num_iterations = 10  # Number of iterations for averaging

    # Main measurement runs
    for i in range(num_iterations):
        with deeparuco_timer("total"):
            result = process_markers(pic.copy(), detector, exp, args, refine_corners, decode_markers)

    # Print DeepAruco++ timing
    deeparuco_timer.print_all("DeepAruco++ execution time")

    # OpenCV ArUco detection timing comparison
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Main measurement runs
    for i in range(num_iterations):
        with opencv_timer("total"):
            corners, ids, rejected = detector.detectMarkers(pic)

    # Print OpenCV timing
    opencv_timer.print_all("OpenCV execution time")

    # Calculate and print speedup factor
    deeparuco_total = deeparuco_timer.get_total_time() / num_iterations
    opencv_total = opencv_timer.get_total_time() / num_iterations
    print(f"\nSpeedup factor: {opencv_total/deeparuco_total:.2f}x")

    # Save the final result
    if result is not None:
        cv2.imwrite(args.out_path, result)
