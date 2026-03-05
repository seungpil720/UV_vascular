import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology
import networkx as nx
import sknw
from scipy import ndimage
from typing import Tuple, Dict, Any

def analyze_octa_network(image_path: str, visualize: bool = True) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Analyzes an OCTA microvascular image to extract network topology and vessel thickness.
    
    Args:
        image_path (str): The file path to the input OCTA image.
        visualize (bool): If True, generates and displays a matplotlib visualization of the network.
        
    Returns:
        Tuple[nx.Graph, Dict[str, Any]]: A tuple containing the extracted NetworkX graph 
                                         and a dictionary of quantitative metrics.
    """
    # 1. Load and Preprocess the Image
    image = io.imread(image_path)
    
    # Standardize to grayscale
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # Handle RGBA
            image = color.rgba2rgb(image)
        image = color.rgb2gray(image)
          
    # 2. Vessel Enhancement (Frangi Filter)
    enhanced_image = filters.frangi(image)
    
    # 3. Binarization (Otsu's Thresholding)
    thresh = filters.threshold_otsu(enhanced_image)
    binary_image = enhanced_image > thresh
    
    # 4. Skeletonization
    skeleton = morphology.skeletonize(binary_image)
    
    # 5. Vessel Thickness Calculation (Euclidean Distance Transform)
    distance_map = ndimage.distance_transform_edt(binary_image)
    skeleton_radii = distance_map[skeleton]
    
    avg_thickness_pixels = np.mean(skeleton_radii) * 2
    max_thickness = np.max(skeleton_radii) * 2
    median_thickness = np.median(skeleton_radii) * 2

    # 6. Graph Extraction
    graph = sknw.build_sknw(skeleton)
    
    # 7. Quantitative Network Metrics
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    
    degrees = [deg for node, deg in graph.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    
    # Robustly calculate total vessel length
    total_length = sum(data.get('weight', 0) for u, v, data in graph.edges(data=True))
    
    # Compile metrics dictionary for programmatic use
    metrics = {
        "file_name": image_path,
        "node_count": node_count,
        "edge_count": edge_count,
        "avg_degree": avg_degree,
        "total_vessel_length": total_length,
        "avg_thickness": avg_thickness_pixels,
        "median_thickness": median_thickness,
        "max_thickness": max_thickness
    }
    
    # Console Output
    print(f"File Name: {image_path}")
    print(f"--- Network Analysis Results ---")
    print(f"Total Nodes (Branching Points): {node_count}")
    print(f"Total Edges (Vessel Segments): {edge_count}")
    print(f"Average Degree (Complexity): {avg_degree:.2f}")
    print(f"Total Vessel Length: {total_length:.2f} pixels")
    print(f"Average Vessel Thickness: {avg_thickness_pixels:.2f} pixels")
    print(f"Median Vessel Thickness: {median_thickness:.2f} pixels")
    print(f"Max Vessel Thickness: {max_thickness:.2f} pixels")

    # 8. Visualization
    if visualize:
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
        
        # Draw edges
        for u, v, data in graph.edges(data=True):
            if 'pts' in data:
                ps = data['pts']
                axes[2].plot(ps[:, 1], ps[:, 0], 'green', linewidth=1.5)
        
        # Draw nodes
        nodes = graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        if len(ps) > 0:
            axes[2].plot(ps[:, 1], ps[:, 0], 'r.', markersize=4)
            
        plt.tight_layout()
        plt.show()

    return graph, metrics

if __name__ == "__main__":
    # Example usage when the script is run directly
    # graph, data = analyze_octa_network('path_to_your_image.png')
    pass
