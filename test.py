import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

# --- 1. Setup ---
img = Image.new('RGB', (400, 300), 'black')
out_img = Image.new(img.mode, img.size)
BACKGROUND_COLOR = (0, 0, 0) # Define background color for erasing

# --- 2. State variables ---
current_color = (255, 255, 255) # Drawing/filling color is white
drawn_points = []
current_mode = "draw" # Mode can be "draw" or "erase"

# --- NEW: Manual Scan-line Polygon Fill Function ---
def manual_fill_polygon(image, vertices, color):
    """
    Fills a polygon on a PIL image using a manual scan-line algorithm.
    This gives pixel-by-pixel control for custom logic.
    """
    # Get a pixel access object for efficient writing
    pixels = image.load()
    
    # 1. Find the Y-range of the polygon to avoid scanning the whole image
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)

    # 2. Iterate through each scan-line (each horizontal row of pixels)
    for y in range(min_y, max_y + 1):
        intersections = []
        # 3. Find all intersection points for the current scan-line
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)] # Wrap around to the first vertex

            # Check if the edge crosses the scan-line y
            if (p1[1] <= y < p2[1]) or (p2[1] <= y < p1[1]):
                # Avoid division by zero for horizontal lines
                if p1[1] != p2[1]:
                    # Calculate the x-intersection using linear interpolation
                    x = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                    intersections.append(x)
        
        # 4. Sort the intersections from left to right
        intersections.sort()

        # 5. Fill the pixels between pairs of intersections
        for i in range(0, len(intersections), 2):
            # Ensure we have a valid pair
            if i + 1 < len(intersections):
                x_start = round(intersections[i])
                x_end = round(intersections[i+1])
                
                for x in range(x_start, x_end + 1):
                    # --- CUSTOM LOGIC CAN GO HERE ---
                    # For example, you could create a pattern:
                    # if (x + y) % 2 == 0:
                    #     pixels[x, y] = color
                    # else:
                    #     pixels[x, y] = (128, 128, 128) # A different color
                    
                    # Standard fill logic:
                    pixels[x, y] = color


# --- 3. Create the GUI Window and Widgets ---
root = tk.Tk()
root.title("Polygon Drawer")

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

# MODIFIED: This function now gathers vertices and calls the manual filler
def fill_polygon():
    """Constructs the polygon shape and fills it using the manual algorithm."""
    if len(drawn_points) < 2:
        print("Not enough points to form a polygon to fill. Need at least 2.")
        return

    print("Constructing polygon and calling manual fill...")
    
    image_height = out_img.height
    last_point = drawn_points[-1]
    first_point = drawn_points[0]
    last_ground_point = (last_point[0], image_height - 1)
    first_ground_point = (first_point[0], image_height - 1)
    
    polygon_vertices = []
    polygon_vertices.extend(drawn_points)
    polygon_vertices.append(last_ground_point)
    polygon_vertices.append(first_ground_point)

    # Call our new manual fill function
    manual_fill_polygon(out_img, polygon_vertices, current_color)

    # After drawing, update the Tkinter display.
    new_tk_image = ImageTk.PhotoImage(out_img)
    image_label.config(image=new_tk_image)
    image_label.image = new_tk_image

def draw_lines():
    """Draws lines connecting the points sequentially and to the ground."""
    if not drawn_points:
        print("Not enough points to draw. Need at least 1.")
        return
    print(f"Drawing lines for {len(drawn_points)} points...")
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

# --- Create a Frame to hold the buttons ---
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
    pixels = out_img.load()
    if current_mode == "draw":
        drawn_points.append((x, y))
        pixels[x, y] = current_color
        image_was_modified = True
    elif current_mode == "erase":
        erase_radius = 3
        points_found_to_erase = []
        for point in drawn_points[:]:
            px, py = point
            if (x - erase_radius <= px <= x + erase_radius) and \
               (y - erase_radius <= py <= y + erase_radius):
                drawn_points.remove(point)
                pixels[px, py] = BACKGROUND_COLOR
                image_was_modified = True
                points_found_to_erase.append(point)
        if image_was_modified: print(f"Erased {len(points_found_to_erase)} point(s) near ({x}, {y})")
    if image_was_modified:
        new_tk_image = ImageTk.PhotoImage(out_img)
        image_label.config(image=new_tk_image)
        image_label.image = new_tk_image

# --- 5. Bind the click event ---
image_label.bind("<Button-1>", on_image_click)

# --- 6. Run the Application ---
print("Tkinter window is running... (Press Ctrl+C in this terminal to exit)")
set_mode_draw()
try:
    root.mainloop()
except KeyboardInterrupt:
    print("\nProgram terminated by user (Ctrl+C).")

# --- 7. Final actions ---
try:
    out_img.save("output_image_polygon_manual.png")
    print("\nModified image saved as 'output_image_polygon_manual.png'")
except Exception as e:
    print(f"Could not save the image. Error: {e}")
print("\n--- Final list of all points drawn ---\n", drawn_points)