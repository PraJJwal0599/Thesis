
from sklearn.cluster import KMeans
import numpy as np
import cv2
import webcolors  # For precise color naming
from scipy.spatial import distance

# Expanded color database for fashion (including common fashion shades)
COLOR_NAMES = {
    #'black': (0, 0, 0),
    #'gray': (128, 128, 128),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'brown': (150, 75, 0),
    'orange': (255, 165, 0),
    'pink': (255, 192, 203),
    'purple': (128, 0, 128),
    'navy': (0, 0, 128),
    'lime': (0, 255, 0),
    'maroon': (128, 0, 0),
    'coral': (255, 127, 127),
    'teal': (0, 128, 128),
    'olive': (128, 128, 0),
    'beige': (245, 245, 220),
    'turquoise': (64, 224, 208),
    'lavender': (230, 230, 250),
    'gold': (255, 215, 0),
    #'silver': (192, 192, 192)
}

def preprocess_image(image):
    """Preprocess image for robust color detection."""
    # Convert to LAB color space for perceptual uniformity
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Apply histogram equalization to L channel (lightness)
    l, a, b = cv2.split(img_lab)
    l = cv2.equalizeHist(l)
    img_lab = cv2.merge((l, a, b))
    # Convert back to BGR for consistency
    img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

def get_dominant_colors(image, k=3):
    """Extract a palette of k dominant colors using KMeans."""
    # Preprocess image
    img = preprocess_image(image)
    # Resize for efficiency
    img = cv2.resize(img, (64, 64))
    img = img.reshape((-1, 3))
    
    # Filter out near-black/white pixels (background noise)
    img = img[np.sum(img, axis=1) > 30]  # Avoid near-black
    img = img[np.sum(img, axis=1) < 720]  # Avoid near-white
    if len(img) < k:  # Fallback if too few pixels
        return [(0, 0, 0)] * k
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(img)
    
    # Get cluster centers and their frequencies
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]  # Sort by frequency
    dominant_colors = [tuple(map(int, kmeans.cluster_centers_[i])) for i in sorted_indices]
    
    # Ensure k colors are returned, padding with black if needed
    while len(dominant_colors) < k:
        dominant_colors.append((0, 0, 0))
    return dominant_colors[:k]

def get_color_name(rgb_tuple):
    """Match RGB to closest color name, prioritizing fashion-relevant names."""
    try:
        # Try webcolors for precise naming
        color_name = webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        # Fallback to custom color dictionary
        min_dist = float("inf")
        color_name = "unknown"
        for name, rgb in COLOR_NAMES.items():
            dist = distance.euclidean(np.array(rgb), np.array(rgb_tuple))
            if dist < min_dist:
                min_dist = dist
                color_name = name
    return color_name

def get_color_palette(image, k=3):
    """Return a list of color names for the dominant colors."""
    dominant_colors = get_dominant_colors(image, k)
    color_names = [get_color_name(color) for color in dominant_colors]
    # Remove duplicates while preserving order
    seen = set()
    color_names = [x for x in color_names if not (x in seen or seen.add(x))]
    return color_names