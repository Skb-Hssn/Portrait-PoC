import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

# --- 1. Define and load image files ---

# HARDCODE YOUR MAIN IMAGE PATH HERE
file_path = "Imu_1.jpg"
# HARDCODE YOUR TEXTURE IMAGE PATH HERE
texture_file_path = "Imu_2.jpg"

# Load the main background image
try:
    # 'img' will hold the pristine original image for erasing
    img = Image.open(file_path).convert('RGB')
    # 'out_img' is the canvas we will draw on
    out_img = img.copy()
except Exception as e:
    print(f"Error opening main image file '{file_path}': {e}")
    exit()

# Load the texture image to be used for filling
try:
    texture_img = Image.open(texture_file_path).convert('RGB')
    # For performance, get pixel access object and dimensions once
    texture_pixels = texture_img.load()
    texture_width, texture_height = texture_img.size
except Exception as e:
    print(f"Error opening texture image file '{texture_file_path}': {e}")
    exit()


# --- 2. State variables ---
current_color = (255, 255, 255) # Color for points and lines
drawn_points = []
current_mode = "draw"

# --- MODIFIED: Manual Scan-line Polygon Fill Function ---
def manual_fill_polygon(image, vertices, tex_pixels, tex_width, tex_height):
    """
    Fills a polygon by mapping pixels from a texture image.
    """
    pixels = image.load()
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)

    for y in range(min_y, max_y + 1):
        intersections = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            if (p1[1] <= y < p2[1]) or (p2[1] <= y < p1[1]):
                if p1[1] != p2[1]:
                    x = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                    intersections.append(x)
        intersections.sort()
        for i in range(0, len(intersections), 2):
            if i + 1 < len(intersections):
                x_start = round(intersections[i])
                x_end = round(intersections[i+1])
                for x in range(x_start, x_end + 1):
                    # --- THIS IS THE KEY LOGIC CHANGE ---
                    # Use modulo to tile the texture image
                    try:
                        tex_x = (x + 568 - 614)
                        tex_y = (y + 317 - 309)
                        
                        # Get the pixel from the texture and apply it to the main image
                        pixels[x, y] = tex_pixels[tex_x, tex_y]
                    except Exception as e:
                        print("Error: ", tex_x, tex_y)


# --- 3. Create the GUI Window and Widgets ---
root = tk.Tk()
root.title("Image Texturizer")

# --- Button functions ---
def set_mode_draw():
    global current_mode
    current_mode = "draw"
    image_label.config(cursor="arrow")
    print("Mode set to: DRAW")

def set_mode_erase():
    global current_mode
    current_mode = "erase"
    image_label.config(cursor="dotbox")
    print("Mode set to: ERASE")

# MODIFIED: This function now calls the manual filler with texture data
def fill_polygon():
    """Constructs the polygon and fills it using the texture image."""
    if len(drawn_points) < 2:
        print("Not enough points to form a polygon. Need at least 2.")
        return
    print("Filling polygon with texture...")
    image_height = out_img.height
    last_point = drawn_points[-1]
    first_point = drawn_points[0]
    last_ground_point = (last_point[0], image_height - 1)
    first_ground_point = (first_point[0], image_height - 1)
    
    polygon_vertices = []
    polygon_vertices.extend(drawn_points)
    polygon_vertices.append(last_ground_point)
    polygon_vertices.append(first_ground_point)

    # Call the manual fill function with the texture data
    manual_fill_polygon(out_img, polygon_vertices, texture_pixels, texture_width, texture_height)

    new_tk_image = ImageTk.PhotoImage(out_img)
    image_label.config(image=new_tk_image)
    image_label.image = new_tk_image

def draw_lines():
    """Draws lines connecting the points sequentially and to the ground."""
    if not drawn_points: return
    draw = ImageDraw.Draw(out_img)
    image_height = out_img.height
    first_point = drawn_points[0]
    first_ground_point = (first_point[0], image_height - 1)
    draw.line((first_point, first_ground_point), fill=current_color, width=1)
    if len(drawn_points) > 1:
        for i in range(len(drawn_points) - 1):
            draw.line((drawn_points[i], drawn_points[i+1]), fill=current_color, width=1)
        last_point = drawn_points[-1]
        last_ground_point = (last_point[0], image_height - 1)
        draw.line((last_point, last_ground_point), fill=current_color, width=1)
    new_tk_image = ImageTk.PhotoImage(out_img)
    image_label.config(image=new_tk_image)
    image_label.image = new_tk_image

# --- Create UI Frames and Buttons ---
button_frame = tk.Frame(root)
button_frame.pack(pady=5)
draw_button = tk.Button(button_frame, text="Draw Mode", command=set_mode_draw)
draw_button.pack(side=tk.LEFT, padx=10)
erase_button = tk.Button(button_frame, text="Erase Mode", command=set_mode_erase)
erase_button.pack(side=tk.LEFT, padx=10)
fill_button = tk.Button(button_frame, text="Fill Polygon", command=fill_polygon)
fill_button.pack(side=tk.LEFT, padx=10)
draw_lines_button = tk.Button(button_frame, text="Draw Lines", command=draw_lines)
draw_lines_button.pack(side=tk.LEFT, padx=10)

# --- Setup the image display ---
tk_image = ImageTk.PhotoImage(out_img)
image_label = tk.Label(root, image=tk_image)
image_label.pack()

# --- 4. The Click Handler Function ---
def on_image_click(event):
    x, y = event.x, event.y
    image_was_modified = False
    pixels_out = out_img.load()
    if current_mode == "draw":
        drawn_points.append((x, y))
        pixels_out[x, y] = current_color
        image_was_modified = True
    elif current_mode == "erase":
        erase_radius = 3
        for point in drawn_points[:]:
            px, py = point
            if (x - erase_radius <= px <= x + erase_radius) and \
               (y - erase_radius <= py <= y + erase_radius):
                drawn_points.remove(point)
                original_pixel = img.getpixel((px, py))
                pixels_out[px, py] = original_pixel
                image_was_modified = True
    if image_was_modified:
        new_tk_image = ImageTk.PhotoImage(out_img)
        image_label.config(image=new_tk_image)
        image_label.image = new_tk_image

# --- 5. Bind the click event ---
image_label.bind("<Button-1>", on_image_click)

# --- 6. Run the Application ---
print(f"Loaded '{file_path}' with texture '{texture_file_path}'. Window is running...")
set_mode_draw()
try:
    root.mainloop()
except KeyboardInterrupt:
    print("\nProgram terminated by user (Ctrl+C).")

# --- 7. Final actions ---
try:
    out_img.save("output_image_textured.png")
    print("\nModified image saved as 'output_image_textured.png'")
except Exception as e:
    print(f"Could not save the image. Error: {e}")
print("\n--- Final list of all points drawn ---\n", drawn_points)