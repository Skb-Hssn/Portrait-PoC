from PIL import Image, ImageFilter, ImageDraw
import numpy as np

def convolve(image_array, kernel):
    """
    Performs 2D convolution.
    Args:
        image_array (numpy.ndarray): Input grayscale image as a NumPy array.
        kernel (numpy.ndarray): The convolution kernel.
    Returns:
        numpy.ndarray: The convolved image array.
    """
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image_array.shape

    # Calculate padding needed
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    # Create an output array initialized to zeros
    output_array = np.zeros_like(image_array, dtype=float)

    # Pad the input image (e.g., with zeros or by replicating border pixels)
    # Using zero padding for simplicity here
    padded_image = np.pad(image_array, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    # Alternatively, for edge replication:
    # padded_image = np.pad(image_array, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')


    # Perform convolution
    for y in range(image_height):
        for x in range(image_width):
            # Extract the region of interest (ROI) from the padded image
            roi = padded_image[y : y + kernel_height, x : x + kernel_width]
            # Element-wise multiplication and sum
            output_array[y, x] = np.sum(roi * kernel)

    return output_array

def laplacian_of_gaussian(image_path, sigma=1.4, kernel_size=None, zero_crossing_threshold=10):
    """
    Performs edge detection using the Laplacian of Gaussian (LoG) operator
    and zero-crossing detection.

    Args:
        image_path (str): Path to the input image.
        sigma (float): Standard deviation for the Gaussian kernel.
        kernel_size (int, optional): Size of the LoG kernel. If None, it's
                                     estimated based on sigma (e.g., 6*sigma+1).
                                     Must be an odd number.
        zero_crossing_threshold (float): Threshold to consider a zero-crossing
                                          as a significant edge.

    Returns:
        PIL.Image.Image: The edge-detected image (binary), or None if an error occurs.
                         The image is also displayed.
    """
    try:
        # 1. Load Image and Convert to Grayscale
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=float)
        height, width = img_array.shape
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # 2. Create Laplacian of Gaussian (LoG) Kernel
    if kernel_size is None:
        # Estimate kernel size: make it cover +/- 3 sigma
        kernel_size = int(6 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1 # Ensure odd kernel size

    log_kernel = np.zeros((kernel_size, kernel_size), dtype=float)
    center = kernel_size // 2

    for y in range(kernel_size):
        for x in range(kernel_size):
            rel_x = x - center
            rel_y = y - center
            # LoG formula: (x^2 + y^2 - 2*sigma^2) / (sigma^4) * exp(-(x^2 + y^2) / (2*sigma^2))
            # For numerical stability, often a normalized LoG is used, or one that sums to zero.
            # Here's a common form:
            term1 = (rel_x**2 + rel_y**2 - 2 * sigma**2) / (sigma**4)
            term2 = np.exp(-(rel_x**2 + rel_y**2) / (2 * sigma**2))
            log_kernel[y, x] = -(1 / (np.pi * sigma**4)) * term1 * term2
            # A simpler, unnormalized LoG often looks like:
            # log_kernel[y,x] = (rel_x**2 + rel_y**2 - 2*sigma**2) * np.exp(-(rel_x**2 + rel_y**2)/(2*sigma**2))


    # Normalize kernel to sum to approximately zero for better results if not already
    # log_kernel = log_kernel - np.mean(log_kernel) # Simple normalization

    # 3. Convolve Image with LoG Kernel
    print("Applying LoG filter...")
    log_filtered_array = convolve(img_array, log_kernel)

    # 4. Zero-Crossing Detection
    print("Detecting zero-crossings...")
    edge_map_array = np.zeros_like(log_filtered_array, dtype=np.uint8)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            neighbors = [
                log_filtered_array[y-1, x], log_filtered_array[y+1, x], # Vertical
                log_filtered_array[y, x-1], log_filtered_array[y, x+1], # Horizontal
                log_filtered_array[y-1, x-1], log_filtered_array[y+1, x+1], # Diagonal 1
                log_filtered_array[y-1, x+1], log_filtered_array[y+1, x-1]  # Diagonal 2
            ]

            # Check if current pixel is close to zero or has a sign change with neighbors
            # and if the magnitude of change is significant (slope across zero-crossing)
            is_zero_crossing = False
            pixel_val = log_filtered_array[y, x]

            # Check horizontal neighbors
            if (pixel_val * log_filtered_array[y, x-1] < 0 and abs(pixel_val - log_filtered_array[y, x-1]) > zero_crossing_threshold) or \
               (pixel_val * log_filtered_array[y, x+1] < 0 and abs(pixel_val - log_filtered_array[y, x+1]) > zero_crossing_threshold):
                is_zero_crossing = True
            # Check vertical neighbors
            if not is_zero_crossing and \
               ((pixel_val * log_filtered_array[y-1, x] < 0 and abs(pixel_val - log_filtered_array[y-1, x]) > zero_crossing_threshold) or \
               (pixel_val * log_filtered_array[y+1, x] < 0 and abs(pixel_val - log_filtered_array[y+1, x]) > zero_crossing_threshold)):
                is_zero_crossing = True
            # Check diagonal neighbors (optional, can make edges thicker or find more)
            # For simplicity, we'll stick to horizontal/vertical for now, but you can add them:
            # if not is_zero_crossing and \
            #    ((pixel_val * log_filtered_array[y-1, x-1] < 0 and abs(pixel_val - log_filtered_array[y-1, x-1]) > zero_crossing_threshold) or \
            #     (pixel_val * log_filtered_array[y+1, x+1] < 0 and abs(pixel_val - log_filtered_array[y+1, x+1]) > zero_crossing_threshold)):
            #     is_zero_crossing = True
            # if not is_zero_crossing and \
            #    ((pixel_val * log_filtered_array[y-1, x+1] < 0 and abs(pixel_val - log_filtered_array[y-1, x+1]) > zero_crossing_threshold) or \
            #     (pixel_val * log_filtered_array[y+1, x-1] < 0 and abs(pixel_val - log_filtered_array[y+1, x-1]) > zero_crossing_threshold)):
            #     is_zero_crossing = True


            if is_zero_crossing:
                edge_map_array[y, x] = 255  # Mark as edge pixel

    # 5. Convert result to Image and display
    edge_image = Image.fromarray(edge_map_array, 'L')
    edge_image.show(title="LoG Edges (Zero-Crossing)")
    return edge_image

if __name__ == "__main__":
    input_image_path = "Portrait_1.jpg"  # <--- REPLACE WITH YOUR IMAGE PATH

    # For testing, create a dummy image if the input is not found
    try:
        Image.open(input_image_path)
    except FileNotFoundError:
        print(f"'{input_image_path}' not found. Creating a dummy test image.")
        dummy_img = Image.new("L", (200, 150), color=100) # Grayscale background
        draw = ImageDraw.Draw(dummy_img)
        # Draw some shapes with different intensities to create edges
        draw.ellipse((30, 30, 100, 80), fill=200)       # Bright ellipse
        draw.rectangle((80, 70, 170, 120), fill=50)    # Dark rectangle
        draw.line((10,140, 190,10), fill=150, width=3) # Diagonal line
        dummy_img.save(input_image_path)
        print(f"Dummy image saved as '{input_image_path}'")

    print(f"Processing LoG edge detection for '{input_image_path}'...")
    log_edges = laplacian_of_gaussian(
        input_image_path,
        sigma=1.0,                # Adjust sigma for LoG kernel
        kernel_size=7,            # Adjust kernel size (odd number)
        zero_crossing_threshold=8 # Adjust threshold for detecting zero-crossings
    )

    if log_edges:
        print("LoG edge detection successful. Image displayed.")
    else:
        print("LoG edge detection failed.")

    # --- Optional: Compare with a simple Laplacian filter from Pillow ---
    # try:
    #     print("\nApplying Pillow's simple Laplacian filter for comparison:")
    #     img_pil = Image.open(input_image_path).convert('L')
    #     # Apply Gaussian blur first for better results with simple Laplacian
    #     blurred_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=1.0))
    #     laplacian_pil = blurred_pil.filter(ImageFilter.LAPLACIAN)
    #     # The output of Pillow's LAPLACIAN is often centered around 128,
    #     # not directly a zero-crossing map. You'd need to process it further
    #     # for zero-crossings. This just shows the raw filter output.
    #     laplacian_pil.show(title="Pillow Simple Laplacian Output (Not Zero-Crossed)")
    # except Exception as e:
    #     print(f"Error during Pillow Laplacian comparison: {e}")