from PIL import Image

def printHex(pixel):
    print(f'{"".join([f"{channel_value:02x}" for channel_value in pixel])}', end=" ")


def extract_pixels_pillow(image_path):
    try:
        img = Image.open(image_path)
        print(f"Image loaded successfully: {image_path}")
        print(f"Image format: {img.format}, Size: {img.size}, Mode: {img.mode}")

        width, height = img.size

        pixel_data = img.load()
        
        for y in range(0, height):
            for x in range(0, width):
                pixel = pixel_data[x, y]
                
                # printHex(pixel)

            # print()

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    extract_pixels_pillow('Portrait_1.jpg')