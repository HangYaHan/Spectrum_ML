import os
import logging
from datetime import datetime

# Define the default log directory and file
LOG_DIR = os.path.join(os.getcwd(), "logs")
LOG_FILE = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log")

# Ensure the log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def get_logger(name):
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)

# Example usage
if __name__ == "__main__":
    logger = get_logger("example")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
