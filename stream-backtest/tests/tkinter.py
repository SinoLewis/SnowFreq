from tkinter import filedialog
import tkinter as tk

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, "r") as file:
            content = file.read()
            text_box.delete("1.0", tk.END)
            text_box.insert(tk.END, content)

def save_file():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    if file_path:
        content = text_box.get("1.0", tk.END)
        with open(file_path, "w") as file:
            file.write(content)

# Create the main window
window = tk.Tk()
window.title("Text Editor")

# Create the text box
text_box = tk.Text(window)
text_box.pack(fill=tk.BOTH, expand=True)

# Create the menu bar
menu_bar = tk.Menu(window)
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Open", command=open_file)
file_menu.add_command(label="Save", command=save_file)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=window.quit)
menu_bar.add_cascade(label="File", menu=file_menu)
window.config(menu=menu_bar)

# Run the application
window.mainloop()
