import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import re

class ToolTip:
    def __init__(self, widget):
        self.widget = widget
        self.tip_window = None
        self.text = ''

    def show_tip(self, text):
        """Display text in a tooltip window"""
        self.text = text
        if self.tip_window or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + cy + self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))

        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()

def create_tooltip(widget, text):
    tooltip = ToolTip(widget)
    def enter(event):
        tooltip.show_tip(text)
    def leave(event):
        tooltip.hide_tip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)
    return tooltip

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
        self.query_history = []

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

        # Create the QueryBuilder tab
        self.query_builder_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.query_builder_frame, text='QueryBuilder')

        # Create the Plugin3 tab
        self.plugin3_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plugin3_frame, text='Plugin3')

        file_select_frame = ttk.Frame(right_frame)
        file_select_frame.grid(row=0, column=0, pady=10)

        self.solver_button = tk.Button(file_select_frame, text="Select Solver", command=self.select_solver, bg="red", activebackground="red", font=("Helvetica", 12), height=2, width=20)
        self.solver_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.model_button = tk.Button(file_select_frame, text="Select Model", command=self.select_model, bg="red", activebackground="red", font=("Helvetica", 12), height=2, width=20)
        self.model_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        # Add tooltips for solver and model buttons
        self.solver_tooltip = create_tooltip(self.solver_button, "No solver selected")
        self.model_tooltip = create_tooltip(self.model_button, "No model selected")

        query_frame = ttk.Frame(right_frame)
        query_frame.grid(row=1, column=0, pady=10, sticky="nswe")

        self.query_label = ttk.Label(query_frame, text="Query:", style="Large.TLabel")
        self.query_label.grid(row=0, column=0, padx=5, pady=5, sticky='nw')

        self.query_entry = tk.Text(query_frame, width=80, height=20, font=("Helvetica", 12))
        self.query_entry.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')

        button_frame = ttk.Frame(query_frame)
        button_frame.grid(row=2, column=0, pady=10, sticky="ew")

        self.evaluate_button = ttk.Button(button_frame, text="Evaluate", command=self.evaluate_query, style="Large.TButton")
        self.evaluate_button.grid(row=0, column=0, padx=5, pady=5, sticky='e')

        self.query_history_button = ttk.Button(button_frame, text="Query History", command=self.show_query_history, style="Large.TButton")
        self.query_history_button.grid(row=0, column=1, padx=5, pady=5, sticky='e')

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
            self.update_tooltip(self.solver_button, self.solver_tooltip, self.solver_path)
            messagebox.showinfo("Solver Selected", f"Solver path set to: {self.solver_path}")

    def select_model(self):
        self.model_path = filedialog.askopenfilename(title="Select Model")
        if self.model_path:
            self.model_button.config(bg="green", activebackground="green", text="Model Selected")
            self.update_tooltip(self.model_button, self.model_tooltip, self.model_path)
            messagebox.showinfo("Model Selected", f"Model path set to: {self.model_path}")

    def update_tooltip(self, button, tooltip, text):
        button.unbind('<Enter>')
        button.unbind('<Leave>')
        tooltip = create_tooltip(button, text)

    def evaluate_query(self):
        query = self.query_entry.get("1.0", tk.END).strip()
        if not self.solver_path or not self.model_path:
            messagebox.showwarning("Paths Missing", "Please select both solver and model paths.")
            return

        # Add the query to history
        self.query_history.append(query)

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
        elif selected_tab == 'QueryBuilder':
            if not hasattr(self, 'query_builder_initialized'):
                self.initialize_query_builder_tab()
                self.query_builder_initialized = True
        elif selected_tab == 'Plugin3':
            for widget in self.plugin3_frame.winfo_children():
                widget.destroy()
            plugin3_label = ttk.Label(self.plugin3_frame, text="Coming soon...", style="Large.TLabel")
            plugin3_label.pack(expand=True)

    def initialize_query_builder_tab(self):
        query_builder_frame = ttk.Frame(self.query_builder_frame)
        query_builder_frame.pack(fill=tk.BOTH, expand=True)

        # Query construction area
        self.query_text = tk.Text(query_builder_frame, width=80, height=20, font=("Helvetica", 12))
        self.query_text.grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky='nsew')

        # Grammar buttons
        grammar_elements = [
            "true", "false", "0", "1", "?", "(", ")",
            "exists", "for all",
            "subsumed by", "<=", "cons", "SR",
            "or", "and", "implies",
            "not", "generate", "load", "show features", "show classes",
            "relevant"
        ]

        button_frame = ttk.Frame(query_builder_frame)
        button_frame.grid(row=1, column=0, columnspan=4, pady=10, sticky="ew")

        for i, element in enumerate(grammar_elements):
            btn = ttk.Button(button_frame, text=element, command=lambda e=element: self.insert_grammar_element(e))
            btn.grid(row=i // 4, column=i % 4, padx=5, pady=5)

        # Action buttons
        action_frame = ttk.Frame(query_builder_frame)
        action_frame.grid(row=2, column=0, columnspan=4, pady=10, sticky="ew")

        self.copy_button = ttk.Button(action_frame, text="Copy to Clipboard", command=self.copy_query, style="Large.TButton")
        self.copy_button.grid(row=0, column=0, padx=5)

        self.clear_button = ttk.Button(action_frame, text="Clear", command=self.clear_query, style="Large.TButton")
        self.clear_button.grid(row=0, column=1, padx=5)

        query_builder_frame.grid_rowconfigure(0, weight=1)
        query_builder_frame.grid_columnconfigure(0, weight=1)

    def insert_grammar_element(self, element):
        self.query_text.insert(tk.END, f" {element} ")

    def copy_query(self):
        query = self.query_text.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(query)
        messagebox.showinfo("Copied", "The query has been copied to the clipboard.")

    def clear_query(self):
        self.query_text.delete("1.0", tk.END)

    def show_query_history(self):
        history_window = tk.Toplevel(self.root)
        history_window.title("Query History")

        history_frame = ttk.Frame(history_window)
        history_frame.pack(fill=tk.BOTH, expand=True)

        history_listbox = tk.Listbox(history_frame, font=("Helvetica", 12))
        history_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for query in self.query_history:
            history_listbox.insert(tk.END, query)

        def copy_selected_query():
            selected_query = history_listbox.get(tk.ACTIVE)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_query)
            messagebox.showinfo("Copied", "The query has been copied to the clipboard.")

        copy_button = ttk.Button(history_frame, text="Copy Selected Query", command=copy_selected_query, style="Large.TButton")
        copy_button.pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
