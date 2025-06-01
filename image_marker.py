import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import sys

SQUARE_SIZE = 60
SQUARE_OUTLINE_COLOR = "red"
SQUARE_OUTLINE_WIDTH = 2
# CLOSE_DELAY_MS is removed as the window no longer auto-closes on click

class ImageMarkerApp:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Image Click Marker (Continuous Output)")
        self.image_path = image_path

        self.pil_original_image = None
        self.tk_image = None
        self.canvas_image_item = None
        self.current_square_id = None
        self.last_click_coords = None # To store (x, y)
        # self.close_timer_id = None # Removed

        self.canvas = tk.Canvas(root)
        self.canvas.pack(fill="both", expand=True)

        if not self.load_image():
            # Handle case where image couldn't be loaded from path
            error_label = tk.Label(root, text=f"Could not load image: {self.image_path}", fg="red")
            error_label.pack(pady=20)
            # Schedule the error window to close, signaling this with on_manual_close
            self.root.after(3000, self.on_manual_close)
            return # Stop further initialization

        # Bind mouse click event
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.root.geometry(f"{self.pil_original_image.width}x{self.pil_original_image.height}")

        # Handle manual window close (e.g., clicking the 'X' button)
        self.root.protocol("WM_DELETE_WINDOW", self.on_manual_close)

    def load_image(self):
        try:
            self.pil_original_image = Image.open(self.image_path)
            self.tk_image = ImageTk.PhotoImage(self.pil_original_image)
            self.canvas.config(width=self.pil_original_image.width, height=self.pil_original_image.height)
            self.canvas_image_item = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
            self.canvas.image = self.tk_image # Keep reference
            return True
        except FileNotFoundError:
            print(f"Error: Image file not found at '{self.image_path}'", file=sys.stderr)
            sys.stderr.flush()
            self.pil_original_image = None
            return False
        except Exception as e:
            print(f"An error occurred while loading the image: {e}", file=sys.stderr)
            sys.stderr.flush()
            self.pil_original_image = None
            return False

    def on_canvas_click(self, event):
        if not self.pil_original_image:
            return

        click_x, click_y = event.x, event.y
        self.last_click_coords = (click_x, click_y) # Store the coordinates

        # Remove previous square
        if self.current_square_id:
            self.canvas.delete(self.current_square_id)

        # Draw new square
        x1 = click_x - SQUARE_SIZE // 2
        y1 = click_y - SQUARE_SIZE // 2
        x2 = click_x + SQUARE_SIZE // 2
        y2 = click_y + SQUARE_SIZE // 2
        self.current_square_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline=SQUARE_OUTLINE_COLOR,
            width=SQUARE_OUTLINE_WIDTH
        )

        # Output coordinates immediately to stdout
        self.output_coordinates()

        # The window no longer closes automatically after a click.
        # The lines scheduling self.root.after() for closing are removed.

    def output_coordinates(self):
        """Prints the last clicked coordinates to standard output."""
        if self.last_click_coords:
            # Print coordinates to standard output
            print(f"X:{self.last_click_coords[0]},Y:{self.last_click_coords[1]}")
            sys.stdout.flush() # Important: ensure output is sent immediately
        else:
            # This should ideally not happen if called after a click event
            print("NO_CLICK_DATA_AVAILABLE")
            sys.stdout.flush()

    def on_manual_close(self):
        """Handles the event when the user manually closes the window."""
        print("WINDOW_CLOSED_MANUALLY")
        sys.stdout.flush() # Ensure this message is sent
        self.root.destroy() # Close the Tkinter application

if __name__ == "__main__":
    root = tk.Tk()
    selected_image_path = None
    app_instance = None # To hold the app instance

    if len(sys.argv) > 1:
        selected_image_path = sys.argv[1]
    else:
        # Fallback to file dialog if no argument is provided
        print("No image path provided via command line. Opening file dialog...", file=sys.stderr)
        sys.stderr.flush()
        selected_image_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=(("PNG files", "*.png"),
                       ("JPEG files", "*.jpg;*.jpeg"),
                       ("GIF files", "*.gif"),
                       ("Bitmap files", "*.bmp"),
                       ("All files", "*.*"))
        )

    if selected_image_path:
        try:
            app_instance = ImageMarkerApp(root, selected_image_path)
            # Only run mainloop if image loading was successful within __init__
            if app_instance.pil_original_image:
                root.mainloop()
            # If image loading failed, __init__ handles its own error window timeout
            # and the script will eventually exit.
        except Exception as e:
            print(f"An unhandled exception occurred: {e}", file=sys.stderr)
            sys.stderr.flush()
            if root.winfo_exists(): # If the window is still there, destroy it
                root.destroy()
    else:
        print("NO_IMAGE_SELECTED") # Signal to parent script or user
        sys.stdout.flush()
        if root.winfo_exists(): # Close the empty Tk window if no file was chosen
           root.destroy()