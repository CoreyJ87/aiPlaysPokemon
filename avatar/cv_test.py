import cv2
import numpy as np

def find_most_complex_region(image_path, window_size=(100, 100)):
    # 1. Load the input image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not open or find the image.")
    
    # 2. Reduce high-frequency noise using Gaussian Blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 3. Apply Canny Edge Detection to get a binary edge map
    # (Pixels are 255 for edges, 0 for background)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    # 4. Use an Integral Image to compute window sums in O(1) time
    # This acts as an ultra-fast sliding window replacement
    integral = cv2.integral(edges)
    
    win_h, win_w = window_size
    img_h, img_w = img.shape
    
    max_edges = -1
    best_top_left = (0, 0)
    
    # 5. Slide the window across the image to find the maximum density
    # Stride can be increased (e.g., step > 1) to speed up large images
    stride = 10 
    for y in range(0, img_h - win_h + 1, stride):
        for x in range(0, img_w - win_w + 1, stride):
            
            # Calculate the sum of edge pixels inside the current window using the integral image
            # Formula: BottomRight - BottomLeft - TopRight + TopTop
            current_sum = (integral[y + win_h, x + win_w] 
                           - integral[y, x + win_w] 
                           - integral[y + win_h, x] 
                           + integral[y, x])
            
            if current_sum > max_edges:
                max_edges = current_sum
                best_top_left = (x, y)
                
    # 6. Define the bounding box of the most complex region
    start_x, start_y = best_top_left
    end_x = start_x + win_w
    end_y = start_y + win_h
    
    # 7. Draw the bounding box on a color copy of the original image for visualization
    color_img = cv2.imread(image_path)
    cv2.rectangle(color_img, (start_x, start_y), (end_x, end_y), (0, 0, 255), 3)
    
    # Display the results
    cv2.imshow("Detected Edges", edges)
    cv2.imshow("Most Complex Region", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return (start_x, start_y, end_x, end_y)

# Execute the function
# bounding_box = find_most_complex_region("your_image.jpg", window_size=(150, 150))

if __name__ == "__main__":
    # Example usage
    image_path = "../screenshot.png"  # Replace with your image path
    bounding_box = find_most_complex_region(image_path, window_size=(150, 150))
    print("Bounding Box of Most Complex Region:", bounding_box)