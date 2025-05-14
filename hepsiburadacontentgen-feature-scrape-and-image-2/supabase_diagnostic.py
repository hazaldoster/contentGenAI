#!/usr/bin/env python3
import requests
import logging
import time
import uuid
import traceback
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration - use values that worked in the diagnostic script
SUPABASE_URL = "https://vsczjwvmkqustdbxyvzo.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZzY3pqd3Zta3F1c3RkYnh5dnpvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDE4NzU5NjQsImV4cCI6MjA1NzQ1MTk2NH0.7tlRgk0sPXHZnmbnvPyOkEHT-ptJMK8BGvINY-5YPds"
BUCKET_NAME = "generations"

# Test video URL
VIDEO_URL = "https://replicate.delivery/xezq/A95HW3ZkSY7fGqTNLOgaK0NZSreJAaZjuUK6WizIkuHHg3qUA/tmp29bswha5.mp4"
CONTENT_TYPE = "video-image"  # Same as used in app.py

def store_in_supabase(url, content_type, type):
    """Same implementation as the fixed store_in_supabase function"""
    logger.info(f"⏳ Starting Supabase storage process for {type} content from URL: {url}")
    try:
        # Download content from original URL
        logger.info(f"🔄 Downloading content from source URL: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            logger.error(f"❌ Failed to download content from {url} - Status code: {response.status_code}")
            return url
        
        content_size = len(response.content)
        logger.info(f"✅ Download successful. Content size: {content_size} bytes")
        
        # Generate a unique filename
        timestamp = int(time.time())
        extension = "mp4" if type == "video" else "jpg"
        filename = f"{type}_{content_type}_{timestamp}.{extension}"
        logger.info(f"📝 Generated filename: {filename}")
        
        # Supabase credentials
        project_ref = "vsczjwvmkqustdbxyvzo"
        anon_key = SUPABASE_ANON_KEY
        logger.info(f"🔑 Using Supabase anonymous key")
        
        bucket_id = BUCKET_NAME
        
        # Upload URL
        upload_url = f"https://{project_ref}.supabase.co/storage/v1/object/{bucket_id}/{filename}"
        content_type_header = "video/mp4" if type == "video" else "image/jpeg"
        logger.info(f"🌐 Upload URL: {upload_url}")
        logger.info(f"📊 Content-Type: {content_type_header}")
        
        # Headers - important to include apikey header
        headers = {
            "apikey": anon_key,
            "Authorization": f"Bearer {anon_key}",
            "Content-Type": content_type_header
        }
        
        # Upload to Supabase
        logger.info(f"⬆️ Starting upload to Supabase via REST API")
        upload_response = requests.post(upload_url, headers=headers, data=response.content)
        
        if upload_response.status_code not in (200, 201):
            logger.error(f"❌ Upload failed with status code: {upload_response.status_code}")
            logger.error(f"❌ Response: {upload_response.text}")
            return url
        
        logger.info(f"✅ Upload successful! Response: {upload_response.text}")
        
        # Generate the public URL for the uploaded content
        supabase_url = f"https://{project_ref}.supabase.co/storage/v1/object/public/{bucket_id}/{filename}"
        logger.info(f"🔗 Generated Supabase URL: {supabase_url}")
        
        # Verify the upload was successful
        try:
            logger.info("🔍 Verifying uploaded content is accessible...")
            verify_response = requests.head(supabase_url, timeout=5)
            if verify_response.status_code == 200:
                logger.info(f"✅ Content verification successful - Status: {verify_response.status_code}")
                logger.info(f"✅ Content stored in Supabase: {supabase_url}")
            else:
                logger.warning(f"⚠️ Content verification returned unexpected status: {verify_response.status_code}")
        except Exception as verify_error:
            logger.warning(f"⚠️ Failed to verify content: {str(verify_error)}")
        
        return supabase_url
        
    except Exception as e:
        logger.error(f"❌ Error storing content in Supabase: {str(e)}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return url  # Return original URL if any error occurs

def test_store_video_in_supabase():
    """Test storing a specific video URL in Supabase"""
    logger.info("=== TESTING STORE_IN_SUPABASE WITH VIDEO URL ===")
    logger.info(f"Video URL: {VIDEO_URL}")
    logger.info(f"Content Type: {CONTENT_TYPE}")
    
    try:
        start_time = time.time()
        result_url = store_in_supabase(VIDEO_URL, CONTENT_TYPE, "video")
        end_time = time.time()
        
        logger.info(f"⏱️ Process took {end_time - start_time:.2f} seconds")
        
        if result_url != VIDEO_URL:
            logger.info(f"✅ SUCCESS: Video successfully uploaded to Supabase")
            logger.info(f"📊 Original URL: {VIDEO_URL}")
            logger.info(f"📊 Supabase URL: {result_url}")
        else:
            logger.error(f"❌ FAILED: Video could not be uploaded to Supabase")
            logger.error(f"📊 URL returned is the same as input: {result_url}")
        
        return result_url
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    logger.info("Starting test script for storing video in Supabase...")
    result = test_store_video_in_supabase()
    
    if result and result != VIDEO_URL:
        logger.info("✅ TEST PASSED: Video was successfully stored in Supabase")
        logger.info(f"🔗 You can access the video at: {result}")
        sys.exit(0)
    else:
        logger.error("❌ TEST FAILED: Video was not successfully stored in Supabase")
        sys.exit(1)
