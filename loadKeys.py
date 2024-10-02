import os
from dotenv import load_dotenv
import openai
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the current working directory
current_dir = Path.cwd()

# Construct the path to the .env file
env_path = current_dir.parent.parent / '.env'

# Check if the .env file exists
if env_path.exists():
    logger.info(f".env file found at: {env_path}")
    # Load environment variables from .env file
    load_dotenv(dotenv_path=env_path)
else:
    logger.error(f".env file not found at: {env_path}")

# Retrieve the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Verify that the API key is set
if openai.api_key:
    logger.info("OpenAI API key loaded successfully.")
else:
    logger.error("Failed to load OpenAI API key.")