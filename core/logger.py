import logging
import sys

def setup_global_logging():
    """
    Configures the root logger for the entire application.
    Call this once at the very beginning of app.py.
    """
    # Create a custom formatter
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Set up the console handler (prints to terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Get the root logger and configure it
    root_logger = logging.getLogger()
    
    # Set the default level to INFO (ignores DEBUG messages to reduce noise)
    root_logger.setLevel(logging.INFO)
    
    # Prevent adding multiple handlers if this runs twice
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)

    # Silence noisy third-party libraries
    logging.getLogger("werkzeug").setLevel(logging.WARNING) # Silences Flask's HTTP spam
    logging.getLogger("httpx").setLevel(logging.WARNING)    # Silences LangChain network spam