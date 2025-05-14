import tkinter as tk
from tkinter import ttk, messagebox, Menu
import joblib
import numpy as np
import os
import webbrowser  # For opening documentation links

class ToolTip:
    """Create a tooltip for any widget"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return
        x = y = 0
        x = self.widget.winfo_pointerx() + 10
        y = self.widget.winfo_pointery() + 10
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, background="#ffffe0", 
                        relief='solid', borderwidth=1, font=("Helvetica", 8))
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        tw = self.tooltip_window
        if tw:
            tw.destroy()
            self.tooltip_window = None

class HeartDiseasePredictor:
    def __init__(self, root):
        # Initialize main window
        self.root = root
        self.root.title('Heart Disease Risk Analyzer')
        self.root.geometry("650x800")
        self.root.minsize(600, 700)
        
        # Configure styles
        self.style = ttk.Style()
        self.configure_styles()
        
        # Load models and scaler
        self.models = self.load_models()
        self.scaler = self.load_scaler()
        
        # Initialize UI
        self.create_menu()
        self.create_widgets()
        self.set_default_values()
        
        # Bind global shortcuts
        self.root.bind("<Control-q>", lambda e: self.root.destroy())
        self.root.bind("<Control-p>", lambda e: self.predict())

    def configure_styles(self):
        """Configure custom UI styles"""
        self.style.configure("Header.TLabel", font=("Helvetica", 10, "bold"))
        self.style.configure("Result.TLabel", padding=10, relief="flat")
        self.style.configure("Accent.TButton", font=("Helvetica", 9, "bold"))
        self.style.configure("Valid.TEntry", fieldbackground="white")
        self.style.configure("Invalid.TEntry", fieldbackground="#ffe6e6")
        self.style.configure("Info.TLabel", foreground="#1a73e8")

    def load_scaler(self):
        """Load and validate scaler"""
        try:
            return joblib.load('scaler.pkl')
        except Exception as e:
            self.show_error("Critical Error", "Failed to load feature scaler", fatal=True)

    def load_models(self):
        """Load all models at startup for faster predictions"""
        model_paths = {
            'BFO': 'Models/bfo_optimize_svm_model.pkl',
            'ACO': 'Models/aco_optimize_svm_model.pkl',
            'DE': 'Models/de_optimize_svm_model.pkl',
            'GA': 'Models/ga_optimize_svm_model.pkl',
            'ABC': 'Models/abc_optimize_svm_model.pkl'
        }
        
        loaded_models = {}
        failed_models = []
        
        for key, path in model_paths.items():
            try:
                loaded_models[key] = joblib.load(path)
            except Exception as e:
                failed_models.append(key)
                messagebox.showwarning(
                    "Model Warning", 
                    f"Could not load {key} model. File might be missing or corrupted."
                )
        
        if not loaded_models:
            self.show_error("Startup Error", "No models could be loaded", fatal=True)
            
        return loaded_models

    def create_menu(self):
        """Create application menu bar"""
        menubar = Menu(self.root)
        
        # File menu
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="New Prediction", command=self.clear_fields, accelerator="Ctrl+Alt+N")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy, accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Help menu
        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="User Guide", command=self.show_help, accelerator="F1")
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Model Info", command=self.show_model_info)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)

    def create_widgets(self):
        """Create and arrange all UI elements"""
        # Input fields section
        input_frame = ttk.LabelFrame(self.root, text="Patient Assessment Parameters", padding=15)
        input_frame.pack(fill="x", padx=15, pady=10)
        
        # Grid layout for input fields
        self.entries = {}
        self.validation_traces = {}  # For real-time validation
        
        for i, (label_text, key, opts, tooltip) in enumerate(self.get_field_definitions()):
            # Label
            label = ttk.Label(input_frame, text=label_text, style="Header.TLabel")
            label.grid(row=i, column=0, padx=5, pady=8, sticky="e")
            
            # Widget container
            widget_frame = ttk.Frame(input_frame)
            widget_frame.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            
            # Create appropriate widget
            if opts:
                combo = ttk.Combobox(widget_frame, values=list(opts.keys()), 
                                   state='readonly', width=25)
                combo.pack(side="left")
                combo.current(0)
                self.entries[key] = (combo, opts)
                
                # Add tooltip
                ToolTip(combo, tooltip)
                
            else:
                entry = ttk.Entry(widget_frame, width=30)
                entry.pack(side="left")
                self.entries[key] = (entry, None)
                
                # Add validation
                self.add_validation(entry)
                
                # Add tooltip
                ToolTip(entry, tooltip)
                
            # Add info icon with tooltip
            info_icon = ttk.Label(widget_frame, text="ⓘ", style="Info.TLabel")
            info_icon.pack(side="left", padx=(5, 0))
            ToolTip(info_icon, tooltip)

        # Model selection
        model_frame = ttk.Frame(self.root, padding=10)
        model_frame.pack(fill="x", padx=15, pady=5)
        
        ttk.Label(model_frame, text="Prediction Model:", style="Header.TLabel").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                       values=list(self.models.keys()), 
                                       state='readonly', width=15)
        self.model_combo.pack(side="left", padx=5)
        if self.models:
            self.model_combo.current(0)
        ToolTip(self.model_combo, "Select optimization algorithm used for prediction")
        
        # Action buttons
        button_frame = ttk.Frame(self.root, padding=10)
        button_frame.pack(fill="x", padx=15, pady=10)
        
        self.predict_btn = ttk.Button(button_frame, text="Analyze Risk", 
                                    command=self.predict, style="Accent.TButton")
        self.predict_btn.pack(side="left", padx=5)
        ToolTip(self.predict_btn, "Run analysis with selected model (Ctrl+P)")
        
        self.clear_btn = ttk.Button(button_frame, text="Clear Fields", 
                                   command=self.clear_fields)
        self.clear_btn.pack(side="left", padx=5)
        ToolTip(self.clear_btn, "Reset all fields to empty (Ctrl+Alt+N)")
        
        # Result display
        self.result_var = tk.StringVar()
        self.result_label = ttk.Label(self.root, textvariable=self.result_var,
                                    wraplength=500, justify="center", padding=15,
                                    relief="groove", style="Result.TLabel")
        self.result_label.pack(fill="x", padx=15, pady=10, ipady=10)
        self.result_var.set("Heart disease risk assessment results will be shown here\n"
                          "Note: This tool provides preliminary analysis only - consult a physician for diagnosis")
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief="sunken")
        self.status_bar.pack(side="bottom", fill="x")

    def get_field_definitions(self):
        """Return enhanced field definitions with tooltips"""
        return [
            ('Age (years)', 'age', None, 'Patient\'s age in years'),
            ('Sex', 'sex', {'Male': 1, 'Female': 0}, 'Select patient gender'),
            ('Chest Pain Type', 'chest pain type', {
                'Typical angina': 1,
                'Atypical angina': 2,
                'Non-anginal pain': 3,
                'Asymptomatic': 4
            }, 'Type of chest pain experienced'),
            ('Resting BP (mm Hg)', 'resting bp s', None, 'Blood pressure at rest (mm Hg)'),
            ('Cholesterol (mg/dl)', 'cholesterol', None, 'Serum cholesterol level'),
            ('Fasting Blood Sugar >120mg/dl', 'fasting blood sugar', 
             {'Yes': 1, 'No': 0}, 'Is fasting blood sugar above 120 mg/dl?'),
            ('Resting ECG', 'resting ecg', {
                'normal': 0,
                'ST-T abnormality': 1,
                'Left ventricular hypertrophy': 2
            }, 'ECG measurement results'),
            ('Max Heart Rate', 'max heart rate', None, 'Maximum heart rate achieved'),
            ('Exercise Angina', 'exercise angina', {'Yes': 1, 'No': 0}, 
             'Does exercise cause angina?'),
            ('Oldpeak', 'oldpeak', None, 'ST depression induced by exercise'),
            ('ST Slope', 'ST slope', {
                'upsloping': 1,
                'flat': 2,
                'downsloping': 3
            }, 'The slope of the peak exercise ST segment'),
        ]

    def add_validation(self, entry):
        """Add real-time validation to entry fields"""
        def validate(*args):
            value = entry_var.get()
            if not value:
                entry.configure(style="Valid.TEntry")
                return
            try:
                float(value)
                entry.configure(style="Valid.TEntry")
                self.status_bar.config(text="Ready")
            except ValueError:
                entry.configure(style="Invalid.TEntry")
                self.status_bar.config(text="Error: Numeric value required")
        
        entry_var = tk.StringVar()
        entry.config(textvariable=entry_var)
        entry_var.trace("w", validate)
        return entry_var

    def set_default_values(self):
        """Set default values for numeric fields"""
        defaults = {
            'age': '50',
            'resting bp s': '120',
            'cholesterol': '200',
            'max heart rate': '150',
            'oldpeak': '1.0'
        }
        for key, value in defaults.items():
            if key in self.entries and isinstance(self.entries[key][0], ttk.Entry):
                self.entries[key][0].delete(0, tk.END)
                self.entries[key][0].insert(0, value)

    def validate_inputs(self):
        """Validate all input fields before prediction"""
        try:
            for key, (widget, opts) in self.entries.items():
                if opts is None:
                    val = widget.get().strip()
                    if not val:
                        raise ValueError(f"{key.capitalize()} cannot be empty")
                    try:
                        float(val)
                    except ValueError:
                        raise ValueError(f"{key.capitalize()} must be a numeric value")
                else:
                    if widget.get() not in opts:
                        raise ValueError(f"Invalid selection for {key}")
            return True
        except ValueError as ve:
            messagebox.showerror("Validation Error", str(ve))
            return False

    def predict(self, event=None):
        """Handle prediction request with enhanced visualization"""
        if not self.validate_inputs():
            return
            
        try:
            # Gather input data
            data = []
            for key, (widget, opts) in self.entries.items():
                if opts:
                    val = opts[widget.get()]
                else:
                    val = float(widget.get())
                data.append(val)
            
            arr = np.array([data])
            X_new = self.scaler.transform(arr)
            
            model_key = self.model_var.get()
            model = self.models[model_key]
            pred = model.predict(X_new)[0]
            
            # Display result with visual feedback
            if pred == 1:
                result_text = (
                    f"Model: {model_key}\n"
                    "⚠️ High Risk of Heart Disease Detected\n"
                    "This is a preliminary analysis suggesting increased risk.\n"
                    "Please consult a cardiologist for comprehensive evaluation."
                )
                self.result_label.configure(foreground="darkred")
            else:
                result_text = (
                    f"Model: {model_key}\n"
                    "✅ Low Risk of Heart Disease Detected\n"
                    "No significant indicators found in this analysis.\n"
                    "Regular checkups are still recommended for preventive care."
                )
                self.result_label.configure(foreground="darkgreen")
            
            self.result_var.set(result_text)
            self.status_bar.config(text="Prediction completed successfully")
            
        except Exception as e:
            self.show_error("Analysis Failed", str(e))

    def clear_fields(self, event=None):
        """Reset all input fields and results"""
        for key, (widget, opts) in self.entries.items():
            if opts is None:
                widget.delete(0, tk.END)  # Clear field without setting defaults
            else:
                widget.current(0)
        
        self.result_var.set(
            "Heart disease risk assessment results will be shown here\n"
            "Note: This tool provides preliminary analysis only - consult a physician for diagnosis"
        )
        self.result_label.configure(foreground="black")
        self.status_bar.config(text="Fields cleared")

    def show_error(self, title, message, fatal=False):
        """Display error messages with appropriate severity"""
        messagebox.showerror(title, message)
        if fatal:
            exit()

    # Enhanced Help functions
    def show_help(self, event=None):
        """Show in-app help documentation"""
        help_text = """
        Heart Disease Risk Analyzer - User Guide
        
        1. Input Fields:
        - Age: Patient's age in years
        - Sex: Male/Female
        - Chest Pain Type: 
          * Typical angina (chest pain typical of heart disease)
          * Atypical angina (unusual chest pain)
          * Non-anginal pain (non-heart-related pain)
          * Asymptomatic (no symptoms)
        
        2. Using the Application:
        - Fill in all patient information
        - Select a prediction model
        - Click 'Analyze Risk' to get results
        - Use 'Clear Fields' to reset all entries
        
        3. Important Notes:
        - This tool provides preliminary analysis only
        - Always consult a qualified physician for diagnosis
        - Results should be interpreted in conjunction with clinical findings
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("User Guide")
        help_window.geometry("600x400")
        
        text_widget = tk.Text(help_window, wrap="word", padx=10, pady=10)
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("1.0", help_text)
        text_widget.configure(state="disabled")
        
        ttk.Button(help_window, text="Close", command=help_window.destroy).pack(pady=5)

    def show_model_info(self):
        """Show detailed model information"""
        info = """
        Optimization Algorithms Used:
        
        1. BFO (Bacterial Foraging Optimization)
           - Inspired by bacterial behavior patterns
           - Good for complex search spaces
        
        2. ACO (Ant Colony Optimization)
           - Inspired by ants' path-finding behavior
           - Effective for combinatorial optimization
        
        3. DE (Differential Evolution)
           - Evolutionary algorithm for global optimization
           - Known for robustness and convergence
        
        4. GA (Genetic Algorithm)
           - Biologically-inspired search heuristic
           - Mimics natural selection process
        
        5. ABC (Artificial Bee Colony)
           - Swarm intelligence-based algorithm
           - Simulates honey bees' foraging behavior
        """
        info_window = tk.Toplevel(self.root)
        info_window.title("Model Information")
        info_window.geometry("500x400")
        
        text_widget = tk.Text(info_window, wrap="word", padx=10, pady=10)
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("1.0", info)
        text_widget.configure(state="disabled")
        
        ttk.Button(info_window, text="Close", command=info_window.destroy).pack(pady=5)

    def show_about(self):
        """Show application information"""
        messagebox.showinfo(
            "About",
            "Heart Disease Risk Analyzer v1.0\n"
            "Using Machine Learning Models for Preliminary Risk Assessment\n"
            "Developed for Medical Professionals\n"
            "© 2023 Medical AI Solutions"
        )

if __name__ == '__main__':
    root = tk.Tk()
    app = HeartDiseasePredictor(root)
    root.mainloop()