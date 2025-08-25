import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

# --- 1. Define and load image files ---

# HARDCODE YOUR MAIN IMAGE PATH HERE
file_path = "frame_00003.png"
# HARDCODE YOUR TEXTURE IMAGE PATH HERE
texture_file_path = "frame_00001.png"

# Load the main background image
try:
    img = Image.open(file_path).convert('RGB')
    out_img = img.copy()
except Exception as e:
    print(f"Error opening main image file '{file_path}': {e}")
    exit()

# Load the texture image
try:
    texture_img = Image.open(texture_file_path).convert('RGB')
    texture_pixels = texture_img.load()
    texture_width, texture_height = texture_img.size
except Exception as e:
    print(f"Error opening texture image file '{texture_file_path}': {e}")
    exit()


# --- 2. State variables ---
current_color = (255, 255, 255) # Color for points and lines
drawn_points = []
current_mode = "draw"

# --- Manual Scan-line Polygon Fill Function ---
def manual_fill_polygon(image, vertices, tex_pixels, tex_width, tex_height):
    """Fills a polygon by mapping pixels from a texture image."""
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
                    # Your custom offset logic
                    try:
                        frame_one_x = 555
                        frame_one_y = 567
                        frame_two_x = 555
                        frame_two_y = 568

                        tex_x = (x + frame_two_x - frame_one_x)
                        tex_y = (y + frame_two_y - frame_one_y)
                        # Ensure texture coordinates are within bounds for this example
                        # If you want tiling, use the modulo operator: tex_x % tex_width
                        if 0 <= tex_x < tex_width and 0 <= tex_y < tex_height:
                           pixels[x, y] = tex_pixels[tex_x, tex_y]
                    except IndexError:
                        # This can happen if offsets go out of bounds. Ignore for now.
                        pass
                    except Exception as e:
                        print(f"Error processing pixel ({x},{y}): {e}")

# --- Helper function to update the canvas image ---
def update_canvas_image():
    """Creates a new PhotoImage and updates the canvas."""
    new_tk_image = ImageTk.PhotoImage(out_img)
    canvas.itemconfig(canvas_image_item, image=new_tk_image)
    # CRITICAL: Keep a reference to the new image object
    canvas.image = new_tk_image

# --- 3. Create the GUI Window and Widgets ---
root = tk.Tk()
root.title("Image Texturizer (Scrollable)")

# --- Button functions ---
def set_mode_draw():
    global current_mode
    current_mode = "draw"
    canvas.config(cursor="arrow")
    print("Mode set to: DRAW")

def set_mode_erase():
    global current_mode
    current_mode = "erase"
    canvas.config(cursor="dotbox")
    print("Mode set to: ERASE")

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

    manual_fill_polygon(out_img, polygon_vertices, texture_pixels, texture_width, texture_height)
    update_canvas_image()

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
    update_canvas_image()

# --- Create UI Frames and Buttons ---
button_frame = tk.Frame(root)
button_frame.pack(side="top", fill="x", pady=5) # Buttons at the top

draw_button = tk.Button(button_frame, text="Draw Mode", command=set_mode_draw)
draw_button.pack(side=tk.LEFT, padx=10)
erase_button = tk.Button(button_frame, text="Erase Mode", command=set_mode_erase)
erase_button.pack(side=tk.LEFT, padx=10)
fill_button = tk.Button(button_frame, text="Fill Polygon", command=fill_polygon)
fill_button.pack(side=tk.LEFT, padx=10)
draw_lines_button = tk.Button(button_frame, text="Draw Lines", command=draw_lines)
draw_lines_button.pack(side=tk.LEFT, padx=10)

# --- NEW: Setup for Scrollable Canvas ---
# A frame will hold the canvas and scrollbars
container_frame = tk.Frame(root)
container_frame.pack(fill="both", expand=True)

canvas = tk.Canvas(container_frame)
v_scrollbar = tk.Scrollbar(container_frame, orient="vertical", command=canvas.yview)
h_scrollbar = tk.Scrollbar(container_frame, orient="horizontal", command=canvas.xview)
canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

canvas.grid(row=0, column=0, sticky="nsew")
v_scrollbar.grid(row=0, column=1, sticky="ns")
h_scrollbar.grid(row=1, column=0, sticky="ew")

container_frame.grid_rowconfigure(0, weight=1)
container_frame.grid_columnconfigure(0, weight=1)

# --- Load image onto the canvas ---
img_width, img_height = out_img.size
tk_image = ImageTk.PhotoImage(out_img)
canvas_image_item = canvas.create_image(0, 0, anchor="nw", image=tk_image)
canvas.image = tk_image # Keep reference
canvas.config(scrollregion=canvas.bbox("all"))

# Set initial window size, capped by screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = min(img_width, screen_width - 50)
window_height = min(img_height, screen_height - 100) # -100 to account for buttons/title bar
root.geometry(f"{window_width}x{window_height}")

# --- 4. The Click Handler Function ---
def on_image_click(event):
    # KEY CHANGE: Convert window coordinates to full canvas coordinates
    x = int(canvas.canvasx(event.x))
    y = int(canvas.canvasy(event.y))
    
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
        update_canvas_image()

# --- 5. Bind the click event ---
canvas.bind("<Button-1>", on_image_click)

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