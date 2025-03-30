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
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.data.data_augment import ValTransform

norm = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)

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
    args = parser.parse_args()

    # Paths
    model_dir = "./models"

    # Load models
    exp = get_exp(None, "yolox-s")
    detector = exp.get_model()
    ckpt = torch.load(f"{model_dir}/{args.detector}.pth", map_location="cpu")
    detector.load_state_dict(ckpt["model"])
    detector.eval()
    
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
    img_info = {"id": 0}
    img_info["file_name"] = args.pic_path
    height, width = pic.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = pic

    # Preprocess image for YOLOX
    preproc = ValTransform(legacy=False)
    img, _ = preproc(img_info, None, exp.input_size)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()

    # Detect markers
    with torch.no_grad():
        outputs = detector(img)
        outputs = postprocess(
            outputs, exp.num_classes, exp.test_conf,
            exp.nmsthre, class_agnostic=True
        )
        detections = outputs[0].cpu()

    # Convert YOLOX detections to same format as before
    xyxy = []
    for det in detections:
        if det is None:
            continue
        x1, y1, x2, y2 = det[:, :4][0].numpy()
        xyxy.append([
            int(max(x1 - (0.2 * (x2 - x1) + 0.5), 0)),
            int(max(y1 - (0.2 * (y2 - y1) + 0.5), 0)),
            int(min(x2 + (0.2 * (x2 - x1) + 0.5), pic.shape[1] - 1)),
            int(min(y2 + (0.2 * (y2 - y1) + 0.5), pic.shape[0] - 1)),
        ])

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
    decoder_out = np.round(decode_markers(np.array(markers)).numpy())
    ids, dists = zip(*[find_id(out) for out in decoder_out])

    # Visualize
    line_width = 2

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

    cv2.imwrite(args.out_path, pic)
