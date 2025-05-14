import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import os
from numpy.fft import fft2, ifft2, fftshift

# output_dir = "test_50px"
# os.makedirs(output_dir, exist_ok=True)
basedir = os.path.join('.', 'maps_cropped_50px')

def extract_dft_feature(image, window_size=15):
    """
    Extract DFT features from image patches
    
    Args:
        image: Input grayscale image
        window_size: Size of the window around keypoint
        
    Returns:
        List of keypoints and their DFT features
    """
    # Parameters for Harris corner detection to find initial keypoints
    sigma = 1.0
    k = 0.05
    threshold = 0.01
    
    # Compute image gradients
    dx = ndimage.sobel(image, axis=0)
    dy = ndimage.sobel(image, axis=1)
    
    # Compute products of derivatives
    dxx = dx * dx
    dxy = dx * dy
    dyy = dy * dy
    
    # Gaussian smoothing
    window = ndimage.gaussian_filter(dxx, sigma)
    dxx_smooth = ndimage.gaussian_filter(dxx, sigma)
    dxy_smooth = ndimage.gaussian_filter(dxy, sigma)
    dyy_smooth = ndimage.gaussian_filter(dyy, sigma)
    
    # Compute Harris response
    det = dxx_smooth * dyy_smooth - dxy_smooth * dxy_smooth
    trace = dxx_smooth + dyy_smooth
    harris_response = det - k * trace * trace
    
    # Threshold and find local maxima
    harris_response[harris_response < threshold * harris_response.max()] = 0
    coords = ndimage.maximum_filter(harris_response, size=10)
    coords = np.column_stack(np.where(coords > 0))
    
    features = []
    half_window = window_size // 2
    
    # Extract DFT features for each keypoint
    for y, x in coords:
        # Check if the window is within image bounds
        if (y >= half_window and y < image.shape[0] - half_window and 
            x >= half_window and x < image.shape[1] - half_window):
            
            # Extract patch
            patch = image[y-half_window:y+half_window+1, 
                          x-half_window:x+half_window+1]
            
            # Compute 2D DFT
            dft = fft2(patch)
            dft_shifted = fftshift(dft)
            
            # Use magnitude spectrum as feature
            magnitude_spectrum = np.log(np.abs(dft_shifted) + 1)
            
            # Flatten the magnitude spectrum to use as feature vector
            feature_vector = magnitude_spectrum.flatten()
            
            # Normalize feature vector
            if np.sum(feature_vector) > 0:
                feature_vector = feature_vector / np.linalg.norm(feature_vector)
                
                features.append({
                    'position': (y, x),
                    'feature': feature_vector
                })
    
    return features

def match_features(features1, features2):
    """
    Match features between two images using Euclidean distance
    
    Args:
        features1: List of features from first image
        features2: List of features from second image
    
    Returns:
        List of matching pairs sorted by similarity score
    """
    matches = []
    
    for i, feat1 in enumerate(features1):
        best_distance = float('inf')
        best_match = None
        
        for j, feat2 in enumerate(features2):
            # Compute Euclidean distance between feature vectors
            distance = np.linalg.norm(feat1['feature'] - feat2['feature'])
            
            if distance < best_distance:
                best_distance = distance
                best_match = j
        
        if best_match is not None:
            matches.append({
                'point1': feat1['position'],
                'point2': features2[best_match]['position'],
                'distance': best_distance
            })
    
    # Sort matches by distance (lower is better)
    matches.sort(key=lambda x: x['distance'])
    
    return matches

def plot_top_matches(image1, image2, matches, n=5):
    """
    Plot the top n matches between two images
    
    Args:
        image1: First input image
        image2: Second input image
        matches: List of matches
        n: Number of top matches to display
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display the images
    ax1.imshow(image1, cmap='gray')
    ax1.set_title('Image 1')
    ax1.axis('off')
    
    ax2.imshow(image2, cmap='gray')
    ax2.set_title('Image 2')
    ax2.axis('off')
    
    # Get top n matches
    top_matches = matches[:n]
    
    # Plot and label the top matches
    for i, match in enumerate(top_matches):
        y1, x1 = match['point1']
        y2, x2 = match['point2']
        
        # Plot markers with rank labels
        ax1.plot(x1, y1, 'ro', markersize=10)
        ax1.text(x1+5, y1+5, f'#{i+1}', color='white', fontsize=12, 
                 bbox=dict(facecolor='red', alpha=0.7))
        
        ax2.plot(x2, y2, 'ro', markersize=10)
        ax2.text(x2+5, y2+5, f'#{i+1}', color='white', fontsize=12,
                 bbox=dict(facecolor='red', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

def main():
    # Load and convert to grayscale
    i = 1
    image1 = np.array(Image.open(os.path.join(basedir, f"a_{i:03d}.png")).convert('L'))
    image2 = np.array(Image.open(os.path.join(basedir, f"b_{i:03d}.png")).convert('L'))
        
    # Extract features from both images
    print("Extracting features from image 1...")
    features1 = extract_dft_feature(image1)
    print(f"Found {len(features1)} features in image 1")
    
    print("Extracting features from image 2...")
    features2 = extract_dft_feature(image2)
    print(f"Found {len(features2)} features in image 2")
    
    # Match features
    print("Matching features...")
    matches = match_features(features1, features2)
    print(f"Found {len(matches)} matches")
    
    # Plot top 5 matches
    print("Plotting top 5 matches...")
    plot_top_matches(image1, image2, matches, n=5)
    
    # Print details of top 5 matches
    print("\nTop 5 matches details:")
    for i, match in enumerate(matches[:5]):
        print(f"Match #{i+1}:")
        print(f"  Image 1 position: {match['point1']}")
        print(f"  Image 2 position: {match['point2']}")
        print(f"  Distance (lower is better): {match['distance']:.4f}")

if __name__ == "__main__":
    main()