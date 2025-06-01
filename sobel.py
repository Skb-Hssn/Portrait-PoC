from PIL import Image
import time
import sys
import math # For sqrt

# Sobel kernels (constants)
SOBEL_X_KERNEL = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]

SOBEL_Y_KERNEL = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
]

# ... (your existing simple_spinner function) ...
def simple_spinner(duration=2, message="Processing"):
    spinner_chars = ['|', '/', '-', '\\']
    end_time = time.time() + duration
    idx = 0
    while time.time() < end_time:
        sys.stdout.write(f"\r{message} {spinner_chars[idx % len(spinner_chars)]} ")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)
    sys.stdout.write(f"\r{message} Done!      \n")
    sys.stdout.flush()


def apply_sobel_pil(image_path, threshold_value=None):
    """
    Applies Sobel edge detection using only Pillow and standard Python.
    Args:
        image_path (str): Path to the input image.
        threshold_value (int, optional): If set (0-255), applies binary thresholding
                                         to the magnitude image.
    Returns:
        tuple: (gradient_x_img, gradient_y_img, magnitude_img, thresholded_img or None)
               Returns (None, None, None, None) if an error occurs.
    """
    try:
        original_img = Image.open(image_path)
        img_gray = original_img.convert('L')  # Convert to grayscale
        pixels_gray = img_gray.load()
        width, height = img_gray.size

        # Output dimensions will be smaller due to kernel size and no padding
        out_width, out_height = width - 2, height - 2
        if out_width <= 0 or out_height <= 0:
            print("Image is too small for Sobel operation (kernel size 3x3).")
            return None, None, None, None

        gx_values = [[0 for _ in range(out_width)] for _ in range(out_height)]
        gy_values = [[0 for _ in range(out_width)] for _ in range(out_height)]
        magnitude_values = [[0 for _ in range(out_width)] for _ in range(out_height)]

        print(f"Applying Sobel to {out_width}x{out_height} region...")
        # Convolve with Sobel kernels
        for y_out in range(out_height): # y for output image
            y_in = y_out + 1 # y for input image (center of kernel)
            if y_out % (out_height // 10 or 1) == 0: # Progress update
                 sys.stdout.write(f"\rSobel progress: {int((y_out/out_height)*100)}% ")
                 sys.stdout.flush()

            for x_out in range(out_width): # x for output image
                x_in = x_out + 1 # x for input image (center of kernel)
                
                gx_sum = 0
                gy_sum = 0
                for ky_idx, ky_row in enumerate(SOBEL_X_KERNEL): # Kernel y
                    for kx_idx, kx_val_gx in enumerate(ky_row):   # Kernel x
                        pixel_val = pixels_gray[x_in + (kx_idx - 1), y_in + (ky_idx - 1)]
                        gx_sum += pixel_val * kx_val_gx
                        gy_sum += pixel_val * SOBEL_Y_KERNEL[ky_idx][kx_idx]
                
                gx_values[y_out][x_out] = gx_sum
                gy_values[y_out][x_out] = gy_sum
                magnitude_values[y_out][x_out] = math.sqrt(gx_sum**2 + gy_sum**2)
        sys.stdout.write(f"\rSobel progress: 100%      \n")
        sys.stdout.flush()


        # --- Helper to normalize and create an image for display ---
        def _normalize_to_image(data_matrix, w, h, is_gradient_component=False):
            flat_data = [val for row in data_matrix for val in row]
            if not flat_data: return Image.new('L', (w, h), 0)

            img_out = Image.new('L', (w, h))
            pixels_out = img_out.load()
            
            min_val = min(flat_data)
            max_val = max(flat_data)

            if max_val == min_val:
                # If all values are the same (e.g., flat region)
                # For magnitude, use the value itself (clamped)
                # For Gx/Gy, 0 gradient should be mid-gray (128)
                fill_val = 128 if is_gradient_component and min_val == 0 else int(max(0, min(255, min_val)))
                for r in range(h):
                    for c in range(w):
                        pixels_out[c, r] = fill_val
                return img_out

            for r_idx in range(h):
                for c_idx in range(w):
                    val = data_matrix[r_idx][c_idx]
                    if is_gradient_component: # Scale Gx/Gy from [-max_abs, max_abs] to [0, 255] with 0 at 128
                        # Find true max absolute value across all Gx or Gy for symmetric scaling
                        true_max_abs = max(abs(min_val), abs(max_val))
                        if true_max_abs == 0: # all zero
                            norm_val = 128
                        else:
                            norm_val = int(((val / true_max_abs) * 127.5) + 127.5)
                    else: # For magnitude (always non-negative)
                        norm_val = int(((val - min_val) / (max_val - min_val)) * 255.0)
                    
                    pixels_out[c_idx, r_idx] = max(0, min(255, norm_val)) # Clamp
            return img_out

        gx_image = _normalize_to_image(gx_values, out_width, out_height, is_gradient_component=True)
        gy_image = _normalize_to_image(gy_values, out_width, out_height, is_gradient_component=True)
        magnitude_image = _normalize_to_image(magnitude_values, out_width, out_height)

        thresholded_image = None
        if threshold_value is not None and 0 <= threshold_value <= 255:
            thresholded_image = Image.new('L', (out_width, out_height))
            pixels_thresh_out = thresholded_image.load()
            # Use the already normalized magnitude_image for thresholding
            pixels_mag_in = magnitude_image.load()
            for r in range(out_height):
                for c in range(out_width):
                    pixels_thresh_out[c, r] = 255 if pixels_mag_in[c, r] >= threshold_value else 0
        
        return gx_image, gy_image, magnitude_image, thresholded_image

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred during Sobel processing: {e}")
        import traceback
        traceback.print_exc() # More detailed error for debugging
        return None, None, None, None


# Your existing function (kept for its original purpose)
def extract_and_print_pixels_with_animation(image_path):
    try:
        img = Image.open(image_path)
        width, height = img.size
        print(f"Image for pixel printing: {image_path}, Size: {width}x{height}")
        simple_spinner(duration=0.5, message="Preparing pixel print")
        print("Starting individual pixel data (first few):")

        pixel_accessor = img.load()
        count = 0
        for y in range(height):
            for x in range(width):
                if count >= 10 : # Print only first 10 for this demo
                    if count == 10: print("... (pixel printing truncated for demo) ...")
                    count+=1
                    continue # skip after 10
                
                p = pixel_accessor[x, y]
                r_val, g_val, b_val = 0, 0, 0
                hex_color_str = ""

                if isinstance(p, tuple): # RGB or RGBA
                    r_val, g_val, b_val = p[0], p[1], p[2]
                    hex_color_str = "".join(f"{channel:02X}" for channel in p)
                elif isinstance(p, int): # Grayscale
                    r_val = g_val = b_val = p
                    hex_color_str = f"{p:02X}{p:02X}{p:02X}"
                
                ansi_bg_color = f"\033[48;2;{r_val};{g_val};{b_val}m"
                ansi_reset = "\033[0m"
                print(f"Pixel ({x},{y}): {hex_color_str} {ansi_bg_color}  {ansi_reset} (Original: {p})")
                count+=1


    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Create a dummy image for testing
    # Make it a bit larger for Sobel to be more meaningful
    dummy_width, dummy_height = 60, 40
    try:
        test_img = Image.new('RGB', (dummy_width, dummy_height), color = 'gray')
        # Add some shapes for edges
        draw = ImageDraw.Draw(test_img)
        draw.rectangle([(10,5), (dummy_width-10, dummy_height-15)], fill='white', outline='black')
        draw.ellipse([(15,10), (dummy_width-15, dummy_height-20)], fill='blue', outline='yellow')
        test_img.save("temp_sobel_test_image.png")
        image_file = "temp_sobel_test_image.png"
        print(f"Created dummy image: {image_file}")
    except NameError: # ImageDraw might not be imported if user only has basic PIL
        try:
            test_img = Image.new('RGB', (dummy_width, dummy_height), color = (128,128,128)) # gray
            for i in range(10, dummy_width-10): test_img.putpixel((i, dummy_height//2), (255,0,0)) # red line
            test_img.save("temp_sobel_test_image.png")
            image_file = "temp_sobel_test_image.png"
            print(f"Created basic dummy image (ImageDraw not found): {image_file}")
        except Exception as e_create:
            print(f"Could not create dummy image: {e_create}")
            image_file = input("Enter path to your image: ")
    except Exception as e:
        print(f"Could not create dummy image with ImageDraw: {e}")
        image_file = input("Enter path to your image: ")


    image_file = "Portrait_1.jpg" 
    # --- 1. Demonstrate your original pixel printing function ---
    print("\n--- Demonstrating Individual Pixel Printing ---")
    extract_and_print_pixels_with_animation(image_file)


    # --- 2. Demonstrate Sobel Edge Detection (using only Pillow) ---
    print("\n\n--- Demonstrating Sobel Edge Detection (Pillow only) ---")
    simple_spinner(duration=0.5, message="Preparing Sobel")
    
    # Apply Sobel (this might take a while for larger images)
    gx_img, gy_img, mag_img, thresh_img = apply_sobel_pil(image_file, threshold_value=50) # Threshold at 50

    if mag_img:
        print("Sobel processing complete. Displaying results...")
        print("Note: Images will open in your default image viewer.")
        
        original_for_show = Image.open(image_file)
        original_for_show.show(title="Original Image")
        
        # Display grayscale version
        img_gray_show = original_for_show.convert('L')
        img_gray_show.show(title="Grayscale Image")

        if gx_img: gx_img.show(title="Sobel Gx (Pillow)")
        if gy_img: gy_img.show(title="Sobel Gy (Pillow)")
        mag_img.show(title="Sobel Magnitude (Pillow) - Edges")
        if thresh_img: thresh_img.show(title="Thresholded Edges (Pillow)")
    else:
        print("Sobel processing failed or image was too small.")