import logging
import tkinter as tk
from tkinter import scrolledtext, ttk
import datetime
from typing import Optional

class LoggerSingleton:
    _instance: Optional['LoggerSingleton'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance
    
    def _initialize_logger(self):
        self.logger = logging.getLogger('AppLogger')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler('app.log')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
class LoggerGUI:
    def __init__(self):
        self.logger_singleton = LoggerSingleton()
        self.root = tk.Tk()
        self.root.title("Logger Interface")
        self.root.geometry("800x600")
        
        self._create_widgets()
        
    def _create_widgets(self):
        # Log level selection
        level_frame = ttk.LabelFrame(self.root, text="Log Level", padding="5")
        level_frame.pack(fill="x", padx=5, pady=5)
        
        self.level_var = tk.StringVar(value="INFO")
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in levels:
            ttk.Radiobutton(level_frame, text=level, value=level, 
                           variable=self.level_var).pack(side="left", padx=5)
        
        # Message input
        input_frame = ttk.LabelFrame(self.root, text="Log Message", padding="5")
        input_frame.pack(fill="x", padx=5, pady=5)
        
        self.message_entry = ttk.Entry(input_frame)
        self.message_entry.pack(fill="x", padx=5, pady=5)
        
        # Log button
        self.log_button = ttk.Button(input_frame, text="Log Message", 
                                   command=self._log_message)
        self.log_button.pack(pady=5)
        
        # Log display
        display_frame = ttk.LabelFrame(self.root, text="Log Output", padding="5")
        display_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.log_display = scrolledtext.ScrolledText(display_frame, wrap=tk.WORD)
        self.log_display.pack(fill="both", expand=True, padx=5, pady=5)
        
    def _log_message(self):
        level = self.level_var.get()
        message = self.message_entry.get()
        
        if message:
            log_method = getattr(self.logger_singleton.logger, level.lower())
            log_method(message)
            
            # Display in GUI
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            display_message = f"{timestamp} - {level} - {message}\n"
            self.log_display.insert(tk.END, display_message)
            self.log_display.see(tk.END)
            
            # Clear input
            self.message_entry.delete(0, tk.END)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    logger_gui = LoggerGUI()
    logger_gui.run()