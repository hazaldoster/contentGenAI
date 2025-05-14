import os
import sys
import logging
import dotenv
from time import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

# Test fal.ai client
logger.info("Testing fal.ai client integration")

try:
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    # Import fal.ai client
    import fal_client
    logger.info("✅ fal_client package imported successfully")
    
    # Get API key from environment
    fal_api_key = os.getenv("FAL_API_KEY")
    if not fal_api_key:
        logger.error("❌ FAL_API_KEY not found in environment variables")
        sys.exit(1)
    logger.info(f"✅ FAL_API_KEY found: {fal_api_key[:8]}...")
    
    # Set environment variables required by fal-client
    os.environ["FAL_KEY"] = fal_api_key
    logger.info("✅ FAL_KEY environment variable set")
    
    # Make a simple request to test connectivity
    logger.info("Testing API connectivity with a simple request...")
    
    # Test with a sample image URL
    sample_image_url = "https://static.vecteezy.com/system/resources/previews/005/857/332/non_2x/funny-portrait-of-cute-corgi-dog-outdoors-free-photo.jpg"
    
    # Test the connection with the iclight-v2 model
    start_time = time()
    result = fal_client.subscribe(
        "fal-ai/iclight-v2",
        {
            "image_url": sample_image_url,
            "prompt": "A test image to verify fal.ai connectivity",
            "image_size": "square_hd"
        }
    )
    request_duration = time() - start_time
    
    logger.info(f"✅ API request successful! Response time: {request_duration:.2f} seconds")
    logger.info(f"Response received with status: {result['status'] if 'status' in result else 'unknown'}")
    
    logger.info("All tests passed! fal.ai client is working properly.")
    
except ImportError as e:
    logger.error(f"❌ Failed to import fal_client: {str(e)}")
    logger.error("Please ensure fal-client is installed with: pip install fal-client")
    sys.exit(1)
except Exception as e:
    logger.error(f"❌ Error during fal.ai client test: {str(e)}")
    sys.exit(1) 