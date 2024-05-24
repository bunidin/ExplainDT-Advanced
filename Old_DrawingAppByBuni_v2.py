import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import re

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Graphical User Interface for REPL Interpreter based on ExplainDT")
        self.canvas_size = 560
        self.pixel_size = 20
        self.grid_size = 28
        self.vector = [0] * (self.grid_size * self.grid_size)

        self.solver_path = None
        self.model_path = None

        self.style = ttk.Style()
        self.style.configure("TButton", padding=10, relief="flat", background="#ccc", foreground="#000")
        self.style.configure("Large.TButton", font=("Helvetica", 12), padding=10)
        self.style.configure("Large.TLabel", font=("Helvetica", 12))
        self.style.configure("Large.TEntry", font=("Helvetica", 12), padding=10)

        self.root.geometry("1400x800")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=2)
        main_frame.grid_rowconfigure(0, weight=1)

        # Create a Notebook (tabbed interface) for the left side
        self.notebook = ttk.Notebook(left_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Create the 28x28DrawingPanel2VectorConverter tab
        self.drawing_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.drawing_frame, text='28x28DrawingPanel2VectorConverter')
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        # Create the Plugin2 tab
        self.plugin2_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plugin2_frame, text='Plugin2')

        # Create the Plugin3 tab
        self.plugin3_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plugin3_frame, text='Plugin3')

        file_select_frame = ttk.Frame(right_frame)
        file_select_frame.grid(row=0, column=0, pady=10)

        self.solver_button = tk.Button(file_select_frame, text="Select Solver", command=self.select_solver, bg="red", activebackground="red", font=("Helvetica", 12), height=2, width=20)
        self.solver_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.model_button = tk.Button(file_select_frame, text="Select Model", command=self.select_model, bg="red", activebackground="red", font=("Helvetica", 12), height=2, width=20)
        self.model_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        query_frame = ttk.Frame(right_frame)
        query_frame.grid(row=1, column=0, pady=10, sticky="nswe")

        self.query_label = ttk.Label(query_frame, text="Query:", style="Large.TLabel")
        self.query_label.grid(row=0, column=0, padx=5, pady=5, sticky='nw')

        self.query_entry = tk.Text(query_frame, width=80, height=20, font=("Helvetica", 12))
        self.query_entry.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')

        self.evaluate_button = ttk.Button(query_frame, text="Evaluate", command=self.evaluate_query, style="Large.TButton")
        self.evaluate_button.grid(row=2, column=0, padx=5, pady=5, sticky='e')

        query_frame.grid_rowconfigure(1, weight=1)
        query_frame.grid_columnconfigure(0, weight=1)

        # Add a frame for the attribution label
        attribution_frame = ttk.Frame(main_frame)
        attribution_frame.grid(row=1, column=0, columnspan=2, sticky="se", padx=10, pady=10)

        self.attribution_label = ttk.Label(attribution_frame, text="Implemented by Buni", style="Large.TLabel")
        self.attribution_label.pack(anchor="se")

        # Initialize the 28x28DrawingPanel2VectorConverter tab content
        self.initialize_drawing_tab()
        
    def initialize_drawing_tab(self):
        self.canvas = tk.Canvas(self.drawing_frame, width=self.canvas_size, height=self.canvas_size, bg='black')
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)

        button_frame = ttk.Frame(self.drawing_frame)
        button_frame.grid(row=1, column=0, pady=10)

        self.generate_button = ttk.Button(button_frame, text="Convert to Vector", command=self.generate_vector, style="Large.TButton")
        self.generate_button.grid(row=0, column=0, padx=5)

        self.reset_button = ttk.Button(button_frame, text="Reset", command=self.reset_canvas, style="Large.TButton")
        self.reset_button.grid(row=0, column=1, padx=5)

    def draw(self, event):
        x, y = event.x, event.y
        x1 = (x // self.pixel_size) * self.pixel_size
        y1 = (y // self.pixel_size) * self.pixel_size
        self.canvas.create_rectangle(x1, y1, x1 + self.pixel_size, y1 + self.pixel_size, fill="white", outline="white")
        self.update_vector(x1, y1)

    def update_vector(self, x, y):
        index = (y // self.pixel_size) * self.grid_size + (x // self.pixel_size)
        self.vector[index] = 1

    def generate_vector(self):
        vector_str = f"[{','.join(map(str, self.vector))}]"
        self.show_vector_window(vector_str)

    def show_vector_window(self, vector_str):
        vector_window = tk.Toplevel(self.root)
        vector_window.title("Generated Vector")

        vector_label = tk.Label(vector_window, text=vector_str, wraplength=500, font=("Helvetica", 12))
        vector_label.pack(pady=10, padx=10)

        copy_button = ttk.Button(vector_window, text="Copy to Clipboard", command=lambda: self.copy_to_clipboard(vector_str), style="Large.TButton")
        copy_button.pack(pady=5)

    def copy_to_clipboard(self, vector_str):
        self.root.clipboard_clear()
        self.root.clipboard_append(vector_str)
        messagebox.showinfo("Copied", "The vector has been copied to the clipboard.")

    def reset_canvas(self):
        self.canvas.delete("all")
        self.vector = [0] * (self.grid_size * self.grid_size)

    def select_solver(self):
        self.solver_path = filedialog.askopenfilename(title="Select Solver")
        if self.solver_path:
            self.solver_button.config(bg="green", activebackground="green", text="Solver Selected")
            messagebox.showinfo("Solver Selected", f"Solver path set to: {self.solver_path}")

    def select_model(self):
        self.model_path = filedialog.askopenfilename(title="Select Model")
        if self.model_path:
            self.model_button.config(bg="green", activebackground="green", text="Model Selected")
            messagebox.showinfo("Model Selected", f"Model path set to: {self.model_path}")

    def evaluate_query(self):
        query = self.query_entry.get("1.0", tk.END).strip()
        if not self.solver_path or not self.model_path:
            messagebox.showwarning("Paths Missing", "Please select both solver and model paths.")
            return

        process = subprocess.Popen(
            ['python3', 'interpreter.py', self.solver_path, self.model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(input=query)

        cleaned_output = self.remove_ansi_codes(stdout if stdout else stderr)

        result_window = tk.Toplevel(self.root)
        result_window.title("Query Result")

        result_label = tk.Label(result_window, text=cleaned_output, wraplength=500, font=("Helvetica", 12))
        result_label.pack(pady=10, padx=10)

    def remove_ansi_codes(self, text):
        ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
        return ansi_escape.sub('', text)

    def on_tab_change(self, event):
        selected_tab = event.widget.tab('current')['text']
        if selected_tab == '28x28DrawingPanel2VectorConverter':
            if not hasattr(self, 'drawing_initialized'):
                self.initialize_drawing_tab()
                self.drawing_initialized = True
            self.canvas.update_idletasks()
        elif selected_tab == 'Plugin2':
            for widget in self.plugin2_frame.winfo_children():
                widget.destroy()
            plugin2_label = ttk.Label(self.plugin2_frame, text="Coming soon...", style="Large.TLabel")
            plugin2_label.pack(expand=True)
        elif selected_tab == 'Plugin3':
            for widget in self.plugin3_frame.winfo_children():
                widget.destroy()
            plugin3_label = ttk.Label(self.plugin3_frame, text="Coming soon...", style="Large.TLabel")
            plugin3_label.pack(expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
