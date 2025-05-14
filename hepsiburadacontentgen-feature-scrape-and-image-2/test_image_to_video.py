#!/usr/bin/env python3
import os
import requests
import json
import time
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SUPABASE_URL = "https://vsczjwvmkqustdbxyvzo.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZzY3pqd3Zta3F1c3RkYnh5dnpvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDE4NzU5NjQsImV4cCI6MjA1NzQ1MTk2NH0.7tlRgk0sPXHZnmbnvPyOkEHT-ptJMK8BGvINY-5YPds"
VIDEO_URL = "https://replicate.delivery/xezq/3ZApNaZh9WZGOlp7qzGwZBweH2e9M3XOMeof8s9B66olfWUlC/tmpi0fttwcr.mp4"

def store_in_supabase():
    """Downloads and uploads the specific video URL to Supabase"""
    url = VIDEO_URL
    content_type = "video/mp4"
    file_type = "video"
    
    logger.info(f"⏳ Starting Supabase Storage process for {url}")
    
    try:
        # Generate a unique filename with timestamp
        timestamp = int(time.time())
        filename = f"{timestamp}-{uuid.uuid4()}.mp4"
        bucket_name = "generations"
        
        logger.info(f"📝 Generated filename: {filename}")
        
        # Download the content from the URL
        logger.info(f"⬇️ Downloading from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        content_data = response.content
        logger.info(f"✅ Download complete: {len(content_data)} bytes")
        
        # Upload to Supabase Storage
        logger.info(f"⬆️ Starting upload to Supabase bucket '{bucket_name}' with key '{filename}'")
        
        upload_headers = {
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Content-Type": content_type
        }
        
        upload_url = f"{SUPABASE_URL}/storage/v1/object/{bucket_name}/{filename}"
        logger.info(f"Upload URL: {upload_url}")
        
        upload_response = requests.post(
            upload_url,
            headers=upload_headers,
            data=content_data
        )
        
        if upload_response.status_code in [200, 201]:
            logger.info(f"✅ Upload successful: {upload_response.text}")
            
            # Construct the public URL for the uploaded file
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket_name}/{filename}"
            logger.info(f"🌍 Public URL: {public_url}")
            
            return {
                "success": True,
                "message": "File successfully uploaded to Supabase Storage",
                "url": public_url,
                "response": upload_response.json()
            }
        else:
            logger.error(f"❌ Upload failed: {upload_response.status_code} - {upload_response.text}")
            return {
                "success": False,
                "message": f"Upload failed with status {upload_response.status_code}",
                "error": upload_response.text
            }
            
    except Exception as e:
        logger.error(f"❌ Error in store_in_supabase: {str(e)}")
        return {
            "success": False,
            "message": f"Error: {str(e)}"
        }

if __name__ == "__main__":
    result = store_in_supabase()
    print(json.dumps(result, indent=2))