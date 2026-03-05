import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology
import networkx as nx
import sknw
from scipy import ndimage

def analyze_octa_network(image_path):
    # 1. Load the Image
    image = io.imread(image_path)
    
    # Check if the image has multiple channels
    if len(image.shape) == 3:
        # If it's an RGBA image (4 channels), convert it to RGB first
        if image.shape[2] == 4:
            image = color.rgba2rgb(image)
        # Now it's safe to convert to grayscale
        image = color.rgb2gray(image)
          
    # 2. Vessel Enhancement (Frangi Filter)
    # The Frangi filter highlights continuous, tube-like structures
    enhanced_image = filters.frangi(image)
    
    # 3. Binarization (Thresholding)
    # Use Otsu's method to automatically find the optimal threshold
    thresh = filters.threshold_otsu(enhanced_image)
    binary_image = enhanced_image > thresh
    
    # 4. Skeletonization
    # Reduce the binary vessels to 1-pixel wide lines
    skeleton = morphology.skeletonize(binary_image)
    
    # --- NEW THICKNESS CALCULATION ---
    # Calculate the distance from every vessel pixel to the nearest background pixel
    distance_map = ndimage.distance_transform_edt(binary_image)
    
    # Extract the radius specifically along the skeletonized centerlines
    skeleton_radii = distance_map[skeleton]
    
    # Calculate average diameter (thickness = radius * 2)
    avg_thickness_pixels = np.mean(skeleton_radii) * 2
    
    # Get max and min thickness for more context (optional but helpful)
    max_thickness = np.max(skeleton_radii) * 2
    median_thickness = np.median(skeleton_radii) * 2
    # ---------------------------------


    # 5. Graph Extraction
    # Convert the skeletonized image into a NetworkX graph
    graph = sknw.build_sknw(skeleton)
    
    # 6. Quantitative Network Metrics
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    
    # Calculate average node degree (branching complexity)
    degrees = [deg for node, deg in graph.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    
    # Calculate total vessel length robustly
    # Unpacks the data dictionary directly to avoid KeyError: 0
    total_length = sum(data.get('weight', 0) for u, v, data in graph.edges(data=True))
    
    # Print Results
    print(f"File Name: {image_path}")
    print(f"--- Network Analysis Results ---")
    print(f"Total Nodes (Branching Points): {node_count}")
    print(f"Total Edges (Vessel Segments): {edge_count}")
    print(f"Average Degree (Complexity): {avg_degree:.2f}")
    print(f"Total Vessel Length: {total_length:.2f} pixels")
    print(f"Average Vessel Thickness: {avg_thickness_pixels:.2f} pixels")
    print(f"Median Vessel Thickness: {median_thickness:.2f} pixels")
    print(f"Max Vessel Thickness: {max_thickness:.2f} pixels")

    # 7. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original OCTA')
    axes[0].axis('off')
    
    axes[1].imshow(skeleton, cmap='gray')
    axes[1].set_title('Skeletonized Vessels')
    axes[1].axis('off')
    
    axes[2].imshow(image, cmap='gray')
    axes[2].set_title('Network Graph Overlay')
    axes[2].axis('off')
    
    # Draw nodes and edges over the original image robustly
    for u, v, data in graph.edges(data=True):
        if 'pts' in data:
            ps = data['pts']
            axes[2].plot(ps[:, 1], ps[:, 0], 'green', linewidth=1.5)
    
    nodes = graph.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])
    if len(ps) > 0:
        axes[2].plot(ps[:, 1], ps[:, 0], 'r.', markersize=4)
        
    plt.tight_layout()
    plt.show()

    return graph
