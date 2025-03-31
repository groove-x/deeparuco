import cv2
import numpy as np
import tensorflow as tf
import time
from impl.aruco import find_id
from impl.heatmaps import pos_from_heatmap
from impl.losses import weighted_loss
from impl.utils import marker_from_corners, ordered_corners
from tensorflow.keras.models import load_model
from ultralytics import YOLO

norm = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)

def process_image(pic, detector, regressor, decoder, threshold=9):
    timings = {}
    
    # Detect markers
    t_start = time.time()
    detections = detector(pic, verbose=False, iou=0.5, conf=0.03)[0].cpu().boxes
    timings['detection'] = time.time() - t_start

    # Expanded bboxes
    t_start = time.time()
    xyxy = [
        [
            int(max(det[0] - (0.2 * (det[2] - det[0]) + 0.5), 0)),
            int(max(det[1] - (0.2 * (det[3] - det[1]) + 0.5), 0)),
            int(min(det[2] + (0.2 * (det[2] - det[0]) + 0.5), pic.shape[1] - 1)),
            int(min(det[3] + (0.2 * (det[3] - det[1]) + 0.5), pic.shape[0] - 1)),
        ]
        for det in [
            [int(val) for val in det.xyxy.cpu().numpy()[0]] for det in detections
        ]
    ]
    timings['bbox_expansion'] = time.time() - t_start

    # Crop and normalize
    t_start = time.time()
    crops_ori = [
        cv2.resize(pic[det[1] : det[3], det[0] : det[2]], (64, 64)) for det in xyxy
    ]
    crops = [norm(crop) for crop in crops_ori]
    timings['crop_normalize'] = time.time() - t_start

    # Refine corners
    t_start = time.time()
    corners = regressor(np.array(crops)).numpy()
    timings['corner_refinement'] = time.time() - t_start

    # Convert to (x, y) pairs
    t_start = time.time()
    corners = [
        [(x, y) for x, y in zip(*pos_from_heatmap(pred, cv2.SimpleBlobDetector_create()))]
        for pred in corners
    ]
    timings['corner_conversion'] = time.time() - t_start

    # Discard detections if less than 4 corners
    t_start = time.time()
    keep = [len(cs) == 4 for cs in corners]
    xyxy, crops_ori, corners = zip(
        *[
            (det, crop, cs)
            for det, crop, cs, k in zip(xyxy, crops_ori, corners, keep)
            if k == True
        ]
    )
    timings['corner_filtering'] = time.time() - t_start

    # Ensure corners are ordered
    t_start = time.time()
    corners = [
        ordered_corners([c[0] for c in cs], [c[1] for c in cs]) for cs in corners
    ]
    timings['corner_ordering'] = time.time() - t_start

    # Extract markers from corners
    t_start = time.time()
    markers = []
    for crop, cs in zip(crops_ori, corners):
        marker = marker_from_corners(crop, cs, 32)
        markers.append(norm(cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)))
    timings['marker_extraction'] = time.time() - t_start

    # Get ids from markers
    t_start = time.time()
    decoder_out = np.round(decoder(np.array(markers)).numpy())
    ids, dists = zip(*[find_id(out) for out in decoder_out])
    timings['marker_decoding'] = time.time() - t_start

    return ids, dists, xyxy, corners, timings

def main():
    # Load models
    model_dir = "./models"
    detector = YOLO(f"{model_dir}/det_luma_bc_s.pt")
    regressor = load_model(
        f"{model_dir}/reg_hmap_8.h5",
        custom_objects={"weighted_loss": weighted_loss},
        compile=False
    )
    decoder = load_model(f"{model_dir}/dec_new.h5", compile=False)

    # Use graph execution for tf models
    @tf.function(reduce_retracing=True)
    def refine_corners(crops):
        return regressor(crops)

    @tf.function(reduce_retracing=True)
    def decode_markers(markers):
        return decoder(markers)

    # Load image
    pic = cv2.imread("gen_image.jpg")
    pic_vis = pic.copy()  # Create a copy for visualization

    # Run multiple times for timing
    n_runs = 1000
    times = []
    step_times = {step: [] for step in [
        'detection', 'bbox_expansion', 'crop_normalize', 'corner_refinement',
        'corner_conversion', 'corner_filtering', 'corner_ordering',
        'marker_extraction', 'marker_decoding'
    ]}
    
    for i in range(n_runs):
        start_time = time.time()
        ids, dists, xyxy, corners, timings = process_image(pic, detector, refine_corners, decode_markers)
        end_time = time.time()
        total_time = end_time - start_time
        times.append(total_time)
        
        # Record individual step times
        for step, t in timings.items():
            step_times[step].append(t)
            
        if i == 0:
            print(f"First run (warmup): {total_time*1000:.1f} ms")
            # Visualize markers after first run
            line_width = 2  # Line width for drawing detections
            for cs, det, id, dist in zip(corners, xyxy, ids, dists):
                # Pack 2-by-2
                cs = [(cs[i], cs[i + 1]) for i in range(0, 8, 2)]
                color = (0, 255, 0)  # Green for good detections
                if dist >= 9:  # Using default threshold
                    color = (0, 0, 255)  # Red for poor detections
                width = det[2] - det[0]
                height = det[3] - det[1]
                for i in range(0, 4):
                    p1 = (int(det[0] + cs[i][0] * width), int(det[1] + cs[i][1] * height))
                    p2 = (int(det[0] + cs[(i + 1) % 4][0] * width),
                          int(det[1] + cs[(i + 1) % 4][1] * height))
                    pic_vis = cv2.line(pic_vis, p1, p2, color, line_width, cv2.LINE_AA)
                # Add ID label
                cv2.putText(pic_vis, f"ID: {id}", (det[0], det[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Save visualization
            cv2.imwrite("output_visualization.jpg", pic_vis)
            print("Saved visualization to output_visualization.jpg")
        elif i < 10:  # Print first 10 runs after warmup
            print(f"Run {i+1}: {total_time*1000:.1f} ms")

    # Calculate statistics (excluding first run)
    mean_time = np.mean(times[1:])
    std_time = np.std(times[1:])
    
    print(f"\nOverall Performance Statistics (excluding warmup):")
    print(f"Mean execution time: {mean_time*1000:.1f} ms")
    print(f"Standard deviation: {std_time*1000:.1f} ms")
    
    print("\nDetailed Step Statistics (excluding warmup):")
    for step, t_list in step_times.items():
        mean_t = np.mean(t_list[1:])
        std_t = np.std(t_list[1:])
        print(f"{step:20s}: mean={mean_t*1000:.1f}ms, std={std_t*1000:.1f}ms")

if __name__ == "__main__":
    main() 