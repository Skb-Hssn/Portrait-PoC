from PIL import Image, ImageFilter, ImageDraw # Added ImageDraw for dummy image
import numpy as np
import math # For math.hypot in gradient magnitude

def canny_edge_detection(image_path, low_threshold_ratio=0.05, high_threshold_ratio=0.15, gaussian_blur_radius=1.4, weak_pixel=75, strong_pixel=255):
    """
    Performs Canny Edge Detection on an image and displays it.

    Args:
        image_path (str): Path to the input image.
        low_threshold_ratio (float): Lower threshold for hysteresis thresholding (ratio of max gradient).
        high_threshold_ratio (float): Higher threshold for hysteresis thresholding (ratio of max gradient).
        gaussian_blur_radius (float): Radius for Gaussian blur for noise reduction.
        weak_pixel (int): Value to assign to weak edge pixels (0-255).
        strong_pixel (int): Value to assign to strong edge pixels (0-255).

    Returns:
        PIL.Image.Image: The Canny edge detected image (grayscale), or None if an error occurs.
                         The image is also displayed using img.show().
    """
    try:
        # 1. Load Image and Convert to Grayscale
        img = Image.open(image_path).convert('L')  # 'L' mode for grayscale
        img_array = np.array(img, dtype=float)
        width, height = img.size
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # 2. Noise Reduction (Gaussian Blur)
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_radius))
    blurred_array = np.array(blurred_img, dtype=float)

    # 3. Gradient Calculation (Sobel Operator)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=float)

    gradient_x = np.zeros_like(blurred_array)
    gradient_y = np.zeros_like(blurred_array)
    gradient_magnitude = np.zeros_like(blurred_array)
    gradient_direction = np.zeros_like(blurred_array) # In radians

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            gx = np.sum(blurred_array[y-1:y+2, x-1:x+2] * sobel_x)
            gradient_x[y, x] = gx
            gy = np.sum(blurred_array[y-1:y+2, x-1:x+2] * sobel_y)
            gradient_y[y, x] = gy
            gradient_magnitude[y, x] = math.hypot(gx, gy)
            gradient_direction[y, x] = math.atan2(gy, gx)

    # 4. Non-Maximum Suppression
    nms_image_array = np.zeros_like(gradient_magnitude)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            angle = gradient_direction[y, x] * 180. / math.pi
            angle = (angle + 180) % 180

            q = 255
            r = 255

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = gradient_magnitude[y, x+1]
                r = gradient_magnitude[y, x-1]
            elif (22.5 <= angle < 67.5):
                q = gradient_magnitude[y+1, x-1]
                r = gradient_magnitude[y-1, x+1]
            elif (67.5 <= angle < 112.5):
                q = gradient_magnitude[y+1, x]
                r = gradient_magnitude[y-1, x]
            elif (112.5 <= angle < 157.5):
                q = gradient_magnitude[y-1, x-1]
                r = gradient_magnitude[y+1, x+1]

            if (gradient_magnitude[y, x] >= q) and (gradient_magnitude[y, x] >= r):
                nms_image_array[y, x] = gradient_magnitude[y, x]
            else:
                nms_image_array[y, x] = 0

    # 5. Double Thresholding
    high_threshold = np.max(nms_image_array) * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio

    thresholded_array = np.zeros_like(nms_image_array)
    strong_y, strong_x = np.where(nms_image_array >= high_threshold)
    weak_y, weak_x = np.where((nms_image_array <= high_threshold) & (nms_image_array >= low_threshold))

    thresholded_array[strong_y, strong_x] = strong_pixel
    thresholded_array[weak_y, weak_x] = weak_pixel

    # 6. Edge Tracking by Hysteresis
    final_edges_array = np.copy(thresholded_array)
    # Iterative hysteresis (can be improved for complex cases)
    for _ in range(5): # Iterate a few times
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if final_edges_array[y, x] == weak_pixel:
                    if (final_edges_array[y-1, x-1] == strong_pixel or
                        final_edges_array[y-1, x]   == strong_pixel or
                        final_edges_array[y-1, x+1] == strong_pixel or
                        final_edges_array[y,   x-1] == strong_pixel or
                        final_edges_array[y,   x+1] == strong_pixel or
                        final_edges_array[y+1, x-1] == strong_pixel or
                        final_edges_array[y+1, x]   == strong_pixel or
                        final_edges_array[y+1, x+1] == strong_pixel):
                        final_edges_array[y, x] = strong_pixel
                    else:
                        final_edges_array[y, x] = 0

    canny_image = Image.fromarray(final_edges_array.astype(np.uint8), 'L')
    
    # Display the image
    canny_image.show(title="Canny Edge Detection Result")
    
    return canny_image

if __name__ == "__main__":
    # --- Example Usage ---
    input_image_path = "Portrait_1.jpg"  # <--- REPLACE WITH YOUR IMAGE PATH

    # For testing, create a dummy image if the input is not found
    try:
        Image.open(input_image_path)
    except FileNotFoundError:
        print(f"'{input_image_path}' not found. Creating a dummy test image.")
        dummy_img = Image.new("L", (200, 150), color=128)
        draw = ImageDraw.Draw(dummy_img) # Make sure ImageDraw is imported
        draw.ellipse((30, 30, 100, 80), fill=200, outline=50)
        draw.rectangle((80, 70, 170, 120), fill=50, outline=200)
        dummy_img.save(input_image_path)
        print(f"Dummy image saved as '{input_image_path}'")

    print(f"Processing Canny edge detection for '{input_image_path}'...")
    canny_result_image = canny_edge_detection(
        input_image_path,
        low_threshold_ratio=0.07,
        high_threshold_ratio=0.18,
        gaussian_blur_radius=1.2,
        weak_pixel=50,
        strong_pixel=255
    )

    if canny_result_image:
        print("Canny edge detection successful. Image displayed.")
        # The image is already shown by the function.
        # No need to call .show() again here unless you want to show it a second time.
        # No saving to file here.
    else:
        print("Canny edge detection failed.")

    # The OpenCV comparison part is removed as the focus is on not saving the primary output.