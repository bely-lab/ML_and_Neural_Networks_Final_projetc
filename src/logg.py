import logging
import sys

# Configure a logger for the project
logger = logging.getLogger("MentalHealthModel")
logger.setLevel(logging.INFO)

# Create a stream handler (console output)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create and set a formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)
