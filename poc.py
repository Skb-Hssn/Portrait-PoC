from PIL import Image

import time
import sys
import random # For one of the examples

# --- Animation Functions ---

def simple_spinner(duration=2, message="Processing"):
    """Displays a simple rotating spinner for a given duration."""
    spinner_chars = ['|', '/', '-', '\\']
    end_time = time.time() + duration
    idx = 0
    while time.time() < end_time:
        sys.stdout.write(f"\r{message} {spinner_chars[idx % len(spinner_chars)]} ")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)
    # sys.stdout.write(f"\r{message} Done!      \n") # Clear spinner and add newline
    # sys.stdout.flush()


def extract_pixels_pillow(image_path):
    try:
        # Open the image
        img = Image.open(image_path)
        print(f"Image loaded successfully: {image_path}")
        print(f"Image format: {img.format}, Size: {img.size}, Mode: {img.mode}")

        # Get image dimensions
        width, height = img.size

        # The load() method returns a pixel access object that provides direct access
        # to pixel data. This is much faster for iterating through pixels.
        print("\nExtracting pixels using load():")
        pixel_data = img.load() # This creates a pixel access object
        pixels_load = []
        img_mod = Image.new('RGB', (width, height))
        for y in range(0, height, 4):
            for x in range(0, width, 4):
                pixel = pixel_data[x, y] # [x, y] coordinate
                # pixel_data[x, y] = (0, pixel_data[x, y][1], pixel_data[x, y][2])
                pixels_load.append(pixel)
                img_mod.putpixel((x, y), (0, 0, pixel_data[x, y][2]))
                print(f'{"".join([f"{channel_value:02x}" for channel_value in pixel_data[x, y]])}', end=" ")


                # ansi_bg_color = f"\033[48;2;{pixel_data[x, y][0]};{pixel_data[x, y][1]};{pixel_data[x, y][2]}m"
                # ansi_reset = "\033[0m"
                # print(f"{ansi_bg_color} {ansi_reset}", end="")
            print()
            # simple_spinner(duration=0.1, message="")


        img_mod.save('New_img.png')
        print(f"Total pixels (load): {len(pixels_load)}")
        if pixels_load:
            print(f"First few pixels (load): {pixels_load[:5]}")

        return pixels_load # or pixels_getpixel

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    # simple_spinner()
    all_pixels = extract_pixels_pillow('portrait_cropped.png') # Change to your image file path

    if all_pixels:
        print(f"\n--- Example: First 10 pixels extracted ---")
        for i, p in enumerate(all_pixels[:10]):
            print(f"Pixel {i}: {p}")
        # Note: Storing all pixels in a list like this can consume a lot of memory for large images.
        # Usually, you'd process them as you extract them.