import imreg_dft as ird
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from PIL import Image
from pre_proc import rgb_to_grayscale, negative_film, bg_subtr, contrast_stretching, apply_tukey_window, pad_bg_value


def load_ground_truth(data_dir):
    """Load ground truth data from JSON file"""
    json_path = os.path.join(data_dir, 'ground_truth.json')
    if not os.path.exists(json_path):
        print(f"Warning: No ground_truth.json found in {data_dir}")
        return None
    
    with open(json_path, 'r') as f:
        gt_data = json.load(f)
    
    # Convert to dictionary indexed by pair_id for easy lookup
    gt_dict = {}
    for entry in gt_data['ground_truth']:
        gt_dict[entry['pair_id']] = entry
    
    return gt_dict, gt_data['metadata']


def calculate_errors(predicted, ground_truth):
    """Calculate various error metrics"""
    # Translation errors
    pred_tx, pred_ty = predicted['translation']
    gt_tx, gt_ty = ground_truth['translation']
    
    # Translation error (Euclidean distance)
    translation_error = np.sqrt((pred_tx - gt_tx)**2 + (pred_ty - gt_ty)**2)
    
    # Component-wise translation errors
    tx_error = abs(pred_tx - gt_tx)
    ty_error = abs(pred_ty - gt_ty)
    
    # Rotation error (handle angle wrapping)
    pred_rot = predicted['rotation']
    gt_rot = ground_truth['rotation_degrees']
    
    # Normalize angles to [-180, 180] range
    def normalize_angle(angle):
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    rotation_error = abs(normalize_angle(pred_rot - gt_rot))
    
    return {
        'translation_error': translation_error,
        'tx_error': tx_error,
        'ty_error': ty_error,
        'rotation_error': rotation_error,
        'predicted': predicted,
        'ground_truth': {
            'translation': [gt_tx, gt_ty],
            'rotation': gt_rot
        }
    }

def print_statistics(all_errors, folder_name):
    """Print comprehensive statistics"""
    if not all_errors:
        print(f"No errors to analyze for {folder_name}")
        return
    
    print(f"\n{'='*60}")
    print(f"STATISTICS FOR: {folder_name}")
    print(f"{'='*60}")
    
    # Extract error arrays
    translation_errors = [e['translation_error'] for e in all_errors]
    tx_errors = [e['tx_error'] for e in all_errors]
    ty_errors = [e['ty_error'] for e in all_errors]
    rotation_errors = [e['rotation_error'] for e in all_errors]
    
    # Translation statistics
    print(f"\nTRANSLATION ERRORS (pixels):")
    print(f"  Total Error (Euclidean):")
    print(f"    Mean:   {np.mean(translation_errors):.3f}")
    print(f"    Median: {np.median(translation_errors):.3f}")
    print(f"    Std:    {np.std(translation_errors):.3f}")
    print(f"    Max:    {np.max(translation_errors):.3f}")
    print(f"    Min:    {np.min(translation_errors):.3f}")
    
    print(f"  X-component:")
    print(f"    Mean:   {np.mean(tx_errors):.3f}")
    print(f"    Median: {np.median(tx_errors):.3f}")
    print(f"    Std:    {np.std(tx_errors):.3f}")
    
    print(f"  Y-component:")
    print(f"    Mean:   {np.mean(ty_errors):.3f}")
    print(f"    Median: {np.median(ty_errors):.3f}")
    print(f"    Std:    {np.std(ty_errors):.3f}")
    
    # Rotation statistics
    print(f"\nROTATION ERRORS (degrees):")
    print(f"    Mean:   {np.mean(rotation_errors):.3f}")
    print(f"    Median: {np.median(rotation_errors):.3f}")
    print(f"    Std:    {np.std(rotation_errors):.3f}")
    print(f"    Max:    {np.max(rotation_errors):.3f}")
    print(f"    Min:    {np.min(rotation_errors):.3f}")
    
    # Success rate metrics
    translation_threshold = 2.0  # pixels
    rotation_threshold = 1.0     # degrees
    
    translation_success = sum(1 for e in translation_errors if e <= translation_threshold)
    rotation_success = sum(1 for e in rotation_errors if e <= rotation_threshold)
    both_success = sum(1 for i, _ in enumerate(translation_errors) 
                      if translation_errors[i] <= translation_threshold and 
                         rotation_errors[i] <= rotation_threshold)
    
    total_pairs = len(all_errors)
    print(f"\nSUCCESS RATES:")
    print(f"  Translation (≤{translation_threshold}px): {translation_success}/{total_pairs} ({100*translation_success/total_pairs:.1f}%)")
    print(f"  Rotation (≤{rotation_threshold}°):     {rotation_success}/{total_pairs} ({100*rotation_success/total_pairs:.1f}%)")
    print(f"  Both criteria:        {both_success}/{total_pairs} ({100*both_success/total_pairs:.1f}%)")


def save_detailed_results(all_errors, folder_name, output_dir):
    """Save detailed results to CSV and JSON"""
    if not all_errors:
        return
    
    # CSV format for easy analysis
    csv_path = os.path.join(output_dir, f'{folder_name}_detailed_results.csv')
    with open(csv_path, 'w') as f:
        f.write('pair_id,pred_tx,pred_ty,pred_rot,gt_tx,gt_ty,gt_rot,tx_error,ty_error,translation_error,rotation_error\n')
        for e in all_errors:
            pred = e['predicted']
            gt = e['ground_truth']
            f.write(f"{e['pair_id']},{pred['translation'][0]:.3f},{pred['translation'][1]:.3f},"
                   f"{pred['rotation']:.3f},{gt['translation'][0]:.3f},{gt['translation'][1]:.3f},"
                   f"{gt['rotation']:.3f},{e['tx_error']:.3f},{e['ty_error']:.3f},"
                   f"{e['translation_error']:.3f},{e['rotation_error']:.3f}\n")
    
    print(f"Detailed results saved: {csv_path}")


if __name__ == "__main__":
    DATA_DIR = './data_eval'
    OUTPUT_DIR = './result_eval'
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
    print(f"Found folders: {folders}")

    # Global statistics across all folders
    global_errors = []

    for folder in folders:
        print(f"\nProcessing folder: {folder}")
        in_folder = os.path.join(DATA_DIR, folder)
        out_folder = os.path.join(OUTPUT_DIR, folder)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        # Load ground truth for this folder
        ground_truth, metadata = load_ground_truth(in_folder)
        if ground_truth is None:
            print(f"Skipping {folder} - no ground truth available")
            continue
        
        print(f"Loaded ground truth with {len(ground_truth)} pairs")
        print(f"Generation parameters: {metadata.get('generation_parameters', 'N/A')}")
        
        folder_errors = []
        
        for i in range(1, 11):  # Assuming pairs 1-10
            if i not in ground_truth:
                print(f"Warning: No ground truth for pair {i}")
                continue
                
            try:
                # Load and process images
                im0 = np.array(Image.open(os.path.join(in_folder, f"a_{i:03d}.png")))
                im1 = np.array(Image.open(os.path.join(in_folder, f"b_{i:03d}.png")))

                im0 = rgb_to_grayscale(im0)
                im1 = rgb_to_grayscale(im1)

                im0 = negative_film(im0)
                im1 = negative_film(im1)

                im0 = bg_subtr(im0)
                im1 = bg_subtr(im1)

                im0 = contrast_stretching(im0, 50, 95)
                im1 = contrast_stretching(im1, 50, 95)

                # Run registration
                result = ird.similarity(im0, im1, numiter=3)
                t_y, t_x = result["tvec"]  # translation vector (Y, X)
                rotation = result["angle"]  # degrees

                # Prepare prediction data
                predicted = {
                    'translation': [float(t_x), float(t_y)],
                    'rotation': float(rotation)
                }
                
                # Calculate errors
                errors = calculate_errors(predicted, ground_truth[i])
                errors['pair_id'] = i
                folder_errors.append(errors)
                global_errors.append(errors)
                
                print(f"Pair {i:2d}: Pred({t_x:+6.2f}, {t_y:+6.2f}, {rotation:+6.1f}°) | "
                      f"GT({ground_truth[i]['translation'][0]:+6.2f}, {ground_truth[i]['translation'][1]:+6.2f}, {ground_truth[i]['rotation_degrees']:+6.1f}°) | "
                      f"Errors: {errors['translation_error']:.2f}px, {errors['rotation_error']:.1f}°")

                # Generate visualization if needed
                # if os.environ.get("IMSHOW", "yes") == "yes":
                #     assert "timg" in result
                #     im2 = result['timg']
                #     ird.imshow(im0, im1, im2, cmap='gray', title=f"Pair {i}", subtitle=True)
                #     out_img = os.path.join(out_folder, f"test_pair_{i:03d}.png")
                #     plt.savefig(out_img)
                #     plt.close()
                    
            except Exception as e:
                print(f"Error processing pair {i}: {e}")
                continue
        
        # Generate statistics and plots for this folder
        if folder_errors:
            print_statistics(folder_errors, folder)
            save_detailed_results(folder_errors, folder, out_folder)
    
    # Global statistics across all folders
    if global_errors:
        print_statistics(global_errors, "GLOBAL (All Folders)")