import os
import subprocess
import sys
from PIL import Image, ImageDraw, ImageFilter
from rich.console import Console

import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"'{func.__name__}' executed in: {elapsed_time:.4f} seconds")
        return result
    return wrapper


def printHex(pixel):
    print(f'{"".join([f"{channel_value:02x}" for channel_value in pixel])}', end=" ")


def get_blurred_pixel_value(original_image, width, height, x, y, kernel_size=11):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
    
    if not (0 <= x < width and 0 <= y < height):
        print("Target pixel coordinates are out of bounds.")
        return None

    r_sum, g_sum, b_sum = 0, 0, 0
    num_pixels_in_kernel = 0
    
    # Determine kernel boundaries
    offset = kernel_size // 2

    for ky in range(-offset, offset + 1):      # Kernel y relative to pixel y
        for kx in range(-offset, offset + 1):  # Kernel x relative to pixel x
            current_x = x + kx
            current_y = y + ky

            # Check if the kernel pixel is within image bounds
            if 0 <= current_x < width and 0 <= current_y < height:
                r, g, b = original_image[current_x, current_y]
                r_sum += r
                g_sum += g
                b_sum += b
                num_pixels_in_kernel += 1
    
    if num_pixels_in_kernel == 0: # Should not happen if target (x,y) is valid
        return original_image[x, y] 

    # Calculate average
    avg_r = int(r_sum / num_pixels_in_kernel)
    avg_g = int(g_sum / num_pixels_in_kernel)
    avg_b = int(b_sum / num_pixels_in_kernel)

    return (avg_r, avg_g, avg_b)
    return (0, 0, 0)


import math

def gaussian_kernel(radius, sigma=None):
    """Generate a 1D Gaussian kernel."""
    if sigma is None:
        sigma = radius / 2.0  # rule of thumb
    kernel = [math.exp(-(x**2) / (2 * sigma**2)) for x in range(-radius, radius+1)]
    s = sum(kernel)
    return [v/s for v in kernel]  # normalize


def gaussian_blur(image, radius=2):
    """
    Apply a Gaussian blur to a 2D RGB image using separable convolution.
    
    Args:
        image (list[list[tuple]]): 2D array of pixels (r, g, b).
        radius (int): blur radius.
    
    Returns:
        list[list[tuple]]: blurred image.
    """
    height = len(image)
    width = len(image[0])
    kernel = gaussian_kernel(radius)
    k_len = len(kernel)

    # Horizontal pass
    temp = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            r_total = g_total = b_total = 0.0
            for k in range(k_len):
                dx = k - radius
                nx = min(max(x + dx, 0), width - 1)
                r, g, b = image[y][nx]
                weight = kernel[k]
                r_total += r * weight
                g_total += g * weight
                b_total += b * weight
            temp[y][x] = (r_total, g_total, b_total)

    # Vertical pass
    blurred = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            r_total = g_total = b_total = 0.0
            for k in range(k_len):
                dy = k - radius
                ny = min(max(y + dy, 0), height - 1)
                r, g, b = temp[ny][x]
                weight = kernel[k]
                r_total += r * weight
                g_total += g * weight
                b_total += b * weight
            blurred[y][x] = (int(r_total), int(g_total), int(b_total))

    return blurred


@timing_decorator
def draw_unmatched_pixels(input_image_path, frame_one, first_center_x, first_center_y, frame_two, second_center_x, second_center_y):
    console = Console()
    try:
        img = Image.open(input_image_path)
        console.print(f"[cyan]Loading image: {input_image_path}[/cyan]")
        print(f"Input image format: {img.format}, Size: {img.size}, Mode: {img.mode}")

        width, height = img.size
        
        new_img = Image.new(img.mode, img.size)
        
        pixels_out = new_img.load()

        flag = [[False for _ in range(width)] for _ in range(height)]

        out_img = Image.new(img.mode, img.size)
        out_pixels_out = out_img.load()

        blurred_frame_one = gaussian_blur(frame_one, 5)

        console.print(f"[cyan]Applying modification...[/cyan]")
        for y in range(height):
            for x in range(width):
                sx = (x + second_center_x - first_center_x)
                sy = (y + second_center_y - first_center_y)

                pixels_out[x, y] = frame_one[y][x]

                if sx >= 0 and sx < width and sy >= 0 and sy < height:
                    dif = abs(frame_one[y][x][0] - frame_two[sy][sx][0])
                    + abs(frame_one[y][x][1] - frame_two[sy][sx][1])
                    + abs(frame_one[y][x][2] - frame_two[sy][sx][2])
                    if dif > 0:
                        flag[y][x] = True
                        # out_pixels_out[x, y] = (0, 0, 0)
                        out_pixels_out[x, y] = blurred_frame_one[y][x]
                    else:                    
                        out_pixels_out[x, y] = pixels_out[x, y]

        # for y in range(height):
        #     for x in range(width):
        #         if flag[y][x]:
        #             out_pixels_out[x, y] = get_blurred_pixel_value(pixels_out, width, height, x, y)
                # cnt_tot = 0
                # cnt_blur = 0
                # for i in range(-3, 4, 1):
                #     for j in range(-3, 4, 1):
                #         if 0 <= x + i < width and 0 <= y + j < height:
                #             cnt_tot += 1
                #             if flag[y + j][x + i]:
                #                 cnt_blur += 1
                # if cnt_tot < 2.5 * cnt_blur:
                #     # pass
                #     # out_pixels_out[x, y] = get_blurred_pixel_value(pixels_out, width, height, x, y, kernel_size=19)
                #     out_pixels_out[x, y] = (0, 0, 0)
                # else:
                #     out_pixels_out[x, y] = pixels_out[x, y]
        out_img.show(title="Image Modified")
        
    except FileNotFoundError:
        console.print(f"[red]Error: Input image file not found at {input_image_path}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]An error occurred: {e}[/red]")
        return False


@timing_decorator
def extract_pixels_pillow(image_path):
    try:
        img = Image.open(image_path)
        Console().print(f"[green]Image loaded successfully: {image_path}[/green]")
        print(f"Image format: {img.format}, Size: {img.size}, Mode: {img.mode}")

        width, height = img.size
        pixel_data = img.load()
        pixel_values = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]
        
        for y in range(0, height):
            for x in range(0, width):
                try:
                    pixel = pixel_data[x, y]
                    pixel_values[y][x] = pixel_data[x, y]
                except:
                    # print(x, y)
                    pass

        return pixel_values  

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_coordinates_from_image(image_path_to_process):
    """
    Runs the image_marker_continuous_output.py script, passes the image path,
    and continuously reads coordinates printed to its stdout.
    """
    if not os.path.exists(image_path_to_process):
        print(f"Error: Image file not found at '{image_path_to_process}' in main script.")
        return

    # Path to the image marker script (assuming it's in the same directory)
    marker_script_path = "image_marker.py" # Make sure this filename matches
    # Or provide an absolute path if it's elsewhere:
    # marker_script_path = "/path/to/your/image_marker_continuous_output.py"

    try:
        # Start the subprocess using Popen
        # stdout=subprocess.PIPE allows us to read its output
        # stderr=subprocess.PIPE allows us to read its errors
        # text=True decodes output as strings (Python 3.7+)
        # bufsize=1 enables line buffering, crucial for getting output promptly
        process = subprocess.Popen(
            [sys.executable, marker_script_path, image_path_to_process],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line-buffered
        )

        print(f"Launched image marker script for '{image_path_to_process}'. Click on the image window.")
        print("Waiting for coordinates (or 'WINDOW_CLOSED_MANUALLY')...")

        # Continuously read from the child process's stdout
        # The loop will break when the child's stdout is closed (i.e., child process ends)
        while True:
            output_line = process.stdout.readline() # Read one line
            if not output_line: # If readline returns empty string, pipe is closed
                print("Image marker script stdout closed.")
                break

            line_content = output_line.strip()
            print(f"[Marker Script Output]: {line_content}") # Log what we receive

            if not line_content: # Skip empty lines if any
                continue

            if "WINDOW_CLOSED_MANUALLY" in line_content:
                print("Image marker window was closed by the user.")
                break # Exit the loop as the script has indicated it's done
            elif "NO_IMAGE_SELECTED" in line_content:
                print("Image marker script reported no image was selected.")
                break
            elif line_content.startswith("X:") and ",Y:" in line_content:
                try:
                    parts = line_content.split(',')
                    x_part = parts[0].split(':')[1]
                    y_part = parts[1].split(':')[1]
                    x_coord, y_coord = int(x_part), int(y_part)
                    print(f"RECEIVED COORDINATES in main script: X={x_coord}, Y={y_coord}")
                    return x_coord, y_coord
                    # Here you would do something with x_coord, y_coord
                except (IndexError, ValueError) as e:
                    print(f"Error parsing coordinates from '{line_content}': {e}")
            # else:
                # Could be other messages from the script, already printed by the log line above

        # After the loop, wait for the process to terminate and get return code
        process.wait()
        print(f"Image marker script finished with return code: {process.returncode}")

        # Check for any errors printed to stderr by the child process
        stderr_output = process.stderr.read()
        if stderr_output:
            print(f"--- Marker Script Stderr ---")
            print(stderr_output.strip())
            print(f"----------------------------")

    except FileNotFoundError:
        print(f"Error: The marker script '{marker_script_path}' or python interpreter not found.")
        print("Please ensure python is in your PATH and the script path is correct.")
    except Exception as e:
        print(f"An error occurred while running or interacting with the marker script: {e}")


def draw_square_and_open(image_path, center_x, center_y, square_side=60, outline_color="red", outline_width=3):
    """
    Opens an image, draws a square around a given point, and displays the image.

    Args:
        image_path (str): The path to the image file.
        center_x (int): The x-coordinate of the center of the square.
        center_y (int): The y-coordinate of the center of the square.
        square_side (int, optional): The side length of the square. Defaults to 60.
        outline_color (str, optional): The color of the square's outline. Defaults to "red".
        outline_width (int, optional): The width of the square's outline. Defaults to 3.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # 1. Open the image
        img = Image.open(image_path)
        # Make sure it's in a mode that supports color drawing (e.g., RGB or RGBA)
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')

        # 2. Create a drawing object
        draw = ImageDraw.Draw(img)

        # 3. Calculate square coordinates
        # The given (center_x, center_y) is the center of the square.
        half_side = square_side // 2  # Use integer division

        # Top-left corner
        x1 = center_x - half_side
        y1 = center_y - half_side
        # Bottom-right corner
        x2 = center_x + half_side
        y2 = center_y + half_side

        # 4. Draw the rectangle
        # The coordinates for rectangle are (x0, y0, x1, y1)
        # where (x0, y0) is the top-left and (x1, y1) is the bottom-right.
        draw.rectangle(
            [x1, y1, x2, y2],        # Coordinates as a list or tuple of 4 values
            outline=outline_color,   # Color of the outline
            width=outline_width      # Width of the outline
        )

        # 5. Display the image
        # This will typically open the image in the system's default image viewer.
        # On some systems, it might save to a temporary file and open that.
        img.show(title=f"Image with Square at ({center_x},{center_y})")

        print(f"Displayed image '{image_path}' with a {square_side}x{square_side} square centered at ({center_x}, {center_y}).")
        return True

    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def match(frame_one, frame_two, second_start_x, second_start_y, first_start_x, first_start_y):
    width = len(frame_one[0])
    height = len(frame_one)

    if (first_start_x + 60 >= width or first_start_y + 60 >= height 
        or second_start_x + 60 >= width or second_start_y + 60 >= height
        or first_start_x < 0 or first_start_y < 0
        or second_start_x < 0 or second_start_y < 0):
        return [10000000, 0]
    
    pixel_cnt = 0
    match_cnt = 0

    # print("Startings: ", second_start_x, second_start_y, first_start_x, first_start_y)
    
    for y in range(0, 60 + 1):
        for x in range(0, 60 + 1):
            pixel_cnt += 1
            dif = abs(frame_one[y + first_start_y][x + first_start_x][0] - frame_two[y + second_start_y][x + second_start_x][0]) 
            + abs(frame_one[y + first_start_y][x + first_start_x][1] - frame_two[y + second_start_y][x + second_start_x][1])
            + abs(frame_one[y + first_start_y][x + first_start_x][2] - frame_two[y + second_start_y][x + second_start_x][2])
            
            # printHex(frame_one[y + first_start_y][x + first_start_x])
            # printHex(frame_two[y + second_start_y][x + second_start_x])
            # print()

            if abs(dif) < 15:
                match_cnt += 1
                # return False
            
            if pixel_cnt - match_cnt > 1000:
                return [10000000, 0]

    return [pixel_cnt, match_cnt]


@timing_decorator
def find_position_in_first_image(frame_one, frame_two, start_x, start_y):
    min_val = 1_000_000_000

    yy, xx = -1, -1

    for y in range(start_y - 20, start_y + 20 + 1):
        for x in range(start_x - 40, start_x + 40 + 1):
            [i, j] = match(frame_one, frame_two, x, y, start_x, start_y)
            if min_val > i - j:
                min_val = i - j
                yy, xx = y + 30, x + 30
    print("[Min val, xx, yy] :", min_val, xx, yy)
    return xx, yy


FRAME_2 = 'frame_00001.png'
FRAME_1 = 'output_image_textured.png'

# FRAME_1 = 'Imu_1.jpg'
# FRAME_2 = 'Imu_2.jpg'

if __name__ == "__main__":
    x, y = get_coordinates_from_image(FRAME_1)
    print(x, y)
    frame_one = extract_pixels_pillow(FRAME_1)
    frame_two = extract_pixels_pillow(FRAME_2)

    mx, my = find_position_in_first_image(frame_one, frame_two, x - 30, y - 30)
    print("Match : ", mx, my)

    draw_square_and_open(FRAME_2, mx, my, square_side=60, outline_color="blue", outline_width=2)

    draw_unmatched_pixels(FRAME_1, frame_one, x, y, frame_two, mx, my)
