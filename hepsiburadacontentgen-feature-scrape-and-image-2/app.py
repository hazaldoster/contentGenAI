from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, session, flash
import os
import requests
import json
import time
from openai import OpenAI
import openai
from dotenv import load_dotenv
import uuid
import logging
import socket
import urllib3
import traceback
import sys
from bs4 import BeautifulSoup
from pydantic import BaseModel
from typing import List
import random
import base64
from PIL import Image
import io
import re
from datetime import datetime
from supabase import create_client
import tempfile
import os.path
import replicate
from urllib.parse import quote
from functools import wraps
from werkzeug.utils import secure_filename
import boto3
from io import BytesIO

# Data structure to store generated content
generated_content = []

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log Python version and environment
logger.info(f"Python version: {sys.version}")
logger.info(f"Environment: {os.environ.get('VERCEL_ENV', 'local')}")

# Import fal.ai
try:
    import fal_client
    FAL_CLIENT_AVAILABLE = True
    logger.info("fal.ai client kütüphanesi başarıyla yüklendi.")
except ImportError as e:
    FAL_CLIENT_AVAILABLE = False
    logger.error(f"fal.ai client kütüphanesi yüklenemedi: {str(e)}")

# DNS çözümleme zaman aşımını artır
socket.setdefaulttimeout(30)  # 30 saniye

# Bağlantı havuzu yönetimi
try:
    urllib3.PoolManager(retries=urllib3.Retry(total=5, backoff_factor=0.5))
    logger.info("urllib3 PoolManager başarıyla yapılandırıldı.")
except Exception as e:
    logger.warning(f"urllib3 PoolManager yapılandırılırken hata: {str(e)}")

# Load environment variables
try:
    load_dotenv()
    logger.info("Çevre değişkenleri yüklendi.")
except Exception as e:
    logger.warning(f"Çevre değişkenleri yüklenirken hata: {str(e)}")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key_for_development')

# API anahtarları
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FAL_API_KEY = os.getenv("FAL_API_KEY")
ASTRIA_API_KEY = os.getenv("ASTRIA_API_KEY")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")

# Log API key availability (not the actual keys)
logger.info(f"OPENAI_API_KEY mevcut: {bool(OPENAI_API_KEY)}")
logger.info(f"FAL_API_KEY mevcut: {bool(FAL_API_KEY)}")
logger.info(f"REPLICATE_API_KEY mevcut: {bool(REPLICATE_API_KEY)}")

# Set Replicate API key
if REPLICATE_API_KEY:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY
    logger.info("REPLICATE_API_TOKEN environment variable set successfully")
else:
    # Try to get token directly if it exists
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
    if REPLICATE_API_TOKEN:
        logger.info("Using existing REPLICATE_API_TOKEN")
    else:
        logger.warning("Neither REPLICATE_API_KEY nor REPLICATE_API_TOKEN found in environment variables")

# Fal.ai API yapılandırması - FAL_KEY çevre değişkenini ayarla
if FAL_API_KEY:
    os.environ["FAL_KEY"] = FAL_API_KEY
    logger.info(f"FAL_KEY çevre değişkeni ayarlandı: {FAL_API_KEY[:4]}..." if FAL_API_KEY else "FAL_KEY ayarlanamadı")
else:
    logger.warning("FAL_API_KEY bulunamadı, FAL_KEY çevre değişkeni ayarlanamadı.")

# OpenAI istemcisini yapılandır
client = None
try:
    if OPENAI_API_KEY:
        # OpenAI API anahtarını doğrudan ayarla
        openai.api_key = OPENAI_API_KEY
        
        # OpenAI istemcisini oluştur
        client = OpenAI(
            api_key=OPENAI_API_KEY,
        )
        
        # API bağlantısını test et
        logger.info("OpenAI API bağlantısı test ediliyor...")
        models = client.models.list()
        logger.info(f"OpenAI API bağlantısı başarılı. Kullanılabilir model sayısı: {len(models.data)}")
    else:
        logger.warning("OPENAI_API_KEY bulunamadı, OpenAI istemcisi oluşturulamadı.")
except Exception as e:
    logger.error(f"OpenAI API bağlantısı kurulamadı: {str(e)}")
    logger.error(f"Hata izleme: {traceback.format_exc()}")

# Ensure templates directory exists
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
if not os.path.exists(template_dir):
    logger.warning(f"Templates directory not found at {template_dir}. Creating it.")
    try:
        os.makedirs(template_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create templates directory: {str(e)}")

# Create basic templates if they don't exist
for template_name in ['welcome.html', 'index.html', 'image.html', 'video.html', 'image2.html']:
    template_path = os.path.join(template_dir, template_name)
    if not os.path.exists(template_path):
        logger.warning(f"Template {template_name} not found. Creating a basic version.")
        try:
            with open(template_path, 'w') as f:
                f.write(f"<!DOCTYPE html><html><head><title>{template_name}</title></head><body><h1>{template_name}</h1><p>This is a placeholder template.</p></body></html>")
        except Exception as e:
            logger.error(f"Failed to create template {template_name}: {str(e)}")

def generate_prompt(text: str, feature_type: str, source_type: str = "url") -> dict:
    """
    OpenAI chat completion API kullanarak doğrudan prompt oluşturur.
    IC-Relight v2 modeli için özel olarak tasarlanmış mise-en-scène veya sahne promptları üretir.
    
    Args:
        text: Görsel URL'si veya base64 formatında görsel verisi
        feature_type: "image" veya "video" değeri alabilir
        source_type: "url" (extractedImages için) veya "upload" (imageUpload için)
    """
    if feature_type not in ["image", "video"]:
        raise ValueError("Geçersiz feature_type! 'image' veya 'video' olmalıdır.")
    
    logger.info(f"Prompt oluşturuluyor. Metin: {text[:50]}... Özellik tipi: {feature_type}, Kaynak tipi: {source_type}")
    
    try:
        # Eğer görsel URL'si ise, görseli analiz et
        image_analysis = ""
        if feature_type == "image":
            # Base64 Data URI kontrolü (data:image/... formatı)
            is_base64_image = text.startswith('data:image/')
            
            # Durum 1: Kullanıcı tarafından yüklenen base64 formatındaki görsel
            if is_base64_image or source_type == "upload":
                logger.info(f"{'Base64 formatında görsel' if is_base64_image else 'Yüklenen görsel'} tespit edildi, direk işleniyor")
                try:
                    if is_base64_image:
                        # Base64 kısmını ayır: "data:image/jpeg;base64,/9j/4AAQ..."
                        base64_data = text.split(',', 1)[1] if ',' in text else ''
                        
                        if not base64_data:
                            logger.error("Geçersiz base64 formatı")
                            image_analysis = "Geçersiz base64 formatı. Görsel analizi yapılamadı."
                        else:
                            # Base64'ten görsel oluştur
                            image_data = base64.b64decode(base64_data)
                            image = Image.open(io.BytesIO(image_data))
                    else:
                        # Dosya yüklemesi durumunda, text doğrudan base64 data olarak gelmiş olabilir
                        try:
                            image_data = base64.b64decode(text)
                            image = Image.open(io.BytesIO(image_data))
                        except:
                            # Eğer doğrudan base64 decode edilemiyorsa, text muhtemelen bir URL
                            logger.error("Yüklenen dosya formatı anlaşılamadı")
                            image_analysis = "Görsel formatı anlaşılamadı. Görsel analizi yapılamadı."
                            raise ValueError("Geçersiz dosya formatı")
                    
                    # Görsel boyutlarını kontrol et ve gerekirse yeniden boyutlandır
                    max_size = (1024, 1024)
                    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                        image.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    # Base64 encode et (daha küçük boyut için optimize ederek)
                    output_buffer = io.BytesIO()
                    image.save(output_buffer, format=image.format or 'JPEG', quality=85, optimize=True)
                    image_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
                    
                    # GPT-4 ile görseli analiz et
                    analysis_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": """Analyze this product image in extreme detail:
                                        1. What exactly is this product? Be very specific.
                                        2. Describe its key physical features (size, shape, materials, color).
                                        3. What is the product's context and purpose?
                                        4. Describe the background, lighting and presentation style.
                                        5. List any distinctive design elements or branding.
                                        6. What mood/style does this product convey?"""
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/{image.format.lower() if image.format else 'jpeg'};base64,{image_base64}"
                                        }
                                    }
                                ]
                            }
                        ],
                        temperature=0.3,
                        max_tokens=800
                    )
                    
                    image_analysis = analysis_response.choices[0].message.content
                    logger.info(f"Yüklenen görsel analizi: {image_analysis[:200]}...")
                    
                except Exception as e:
                    logger.error(f"Yüklenen görsel analizi sırasında hata: {str(e)}")
                    logger.error(f"Hata izleme: {traceback.format_exc()}")
                    image_analysis = "Yüklenen görsel analizi yapılamadı."
            
            # Durum 2: URL'den çıkartılan/alınan görseller
            elif source_type == "url" and not is_base64_image:
                try:
                    # Görseli indir
                    logger.info(f"URL'den görsel indiriliyor: {text[:100]}...")
                    response = requests.get(text, timeout=10)
                    if response.status_code != 200:
                        raise ValueError(f"Görsel indirilemedi: HTTP {response.status_code}")
                    
                    # Pillow ile görseli işle
                    image = Image.open(io.BytesIO(response.content))
                    
                    # Görsel boyutlarını kontrol et ve gerekirse yeniden boyutlandır
                    max_size = (1024, 1024)
                    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                        image.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    # Görsel kalitesini optimize et
                    output_buffer = io.BytesIO()
                    image.save(output_buffer, format='JPEG', quality=85, optimize=True)
                    image_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
                    
                    # GPT-4 ile görseli analiz et - ürün görseli üzerinde daha spesifik yönlendirme
                    analysis_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": """Analyze this product image in extreme detail:
                                        1. What exactly is this product? Be very specific.
                                        2. Describe its key physical features (size, shape, materials, color).
                                        3. What is the product's context and purpose?
                                        4. Describe the background, lighting and presentation style.
                                        5. List any distinctive design elements or branding.
                                        6. What mood/style does this product convey?"""
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/{image.format.lower()};base64,{image_base64}"
                                        }
                                    }
                                ]
                            }
                        ],
                        temperature=0.3,
                        max_tokens=800
                    )
                    
                    image_analysis = analysis_response.choices[0].message.content
                    logger.info(f"URL görsel analizi: {image_analysis[:200]}...")
                    logger.info(f"Tam görsel analizi:\n{image_analysis}")
                    
                except Exception as e:
                    logger.error(f"URL görsel analizi sırasında hata: {str(e)}")
                    logger.error(f"Hata izleme: {traceback.format_exc()}")
                    image_analysis = "URL görsel analizi yapılamadı."
            else:
                logger.error("Görsel kaynak türü belirlenemedi veya desteklenmiyor")
                image_analysis = "Görsel kaynak türü belirlenemedi."
        else:
            image_analysis = ""

        # Doğrudan görselden prompt oluştur
        logger.info("Görselden prompt oluşturuluyor...")
        
        # 4 ayrı prompt oluşturmak için tek bir istek gönder
        system_instruction = """You are an expert prompt creator for AI image generation models. Your task is to create exactly 4 different high-quality prompts for IC-Relight v2 image model based on the product image analysis provided.

The prompts will be used to generate improved, marketing-ready images of the product. Each prompt must focus on making the product look professional and attractive while maintaining its identity.

Product Image Analysis:
{image_analysis}

Follow these strict guidelines:
1. Create EXACTLY 4 distinct prompts
2. Each prompt should be 50-100 words
3. Each prompt must be in English
4. Each prompt must maintain the product's original identity and purpose
5. Each prompt must include specific lighting, atmosphere, and setting details
6. Each prompt must focus on making the product look professional and commercial
7. Each prompt must include descriptive details about the product itself
8. Each prompt must include a scene description that serves as a title

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
SCENE1: [Brief scene description]
[Prompt 1]

SCENE2: [Brief scene description]
[Prompt 2]

SCENE3: [Brief scene description]
[Prompt 3] 

SCENE4: [Brief scene description]
[Prompt 4]

Each prompt should have a clearly distinct approach but maintain the product's identity. Do not include any explanations, just provide the 4 formatted prompts."""
        
        prompt_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_instruction.format(image_analysis=image_analysis)},
                {"role": "user", "content": "Create 4 distinct image generation prompts based on the analysis above. Each should present the product in a different professional context or lighting situation."}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Yanıtı işle
        response_text = prompt_response.choices[0].message.content.strip()
        logger.info(f"Prompt yanıtı alındı: {response_text[:100]}...")
        
        # Stil ve promptları ayır
        sections = response_text.split('\n\n')
        
        prompt_data = []
        
        for section in sections:
            lines = section.strip().split('\n')
            if not lines:
                continue
                
            # İlk satırdan sahne açıklamasını çıkar
            scene_line = lines[0]
            if "SCENE" in scene_line.upper() and ":" in scene_line:
                scene = scene_line.split(":", 1)[1].strip()
                # Sahne satırını çıkar ve kalan satırları prompt olarak birleştir
                prompt = " ".join(lines[1:]).strip()
                if prompt and len(prompt) > 10:
                    prompt_data.append({"scene": scene, "prompt": prompt})
        
        # Hiçbir prompt bulunamazsa, alternatif bir format kontrol et (GPT bazen farklı bir format kullanabilir)
        if not prompt_data:
            logger.warning("Standart format bulunamadı, alternatif format kontrol ediliyor...")
            # Farklı formatları deneyebiliriz
            for section in sections:
                if not section.strip():
                    continue
                    
                # Prompt kelimesi geçen satırları ara
                if "PROMPT" in section.upper() or "SCENE" in section.upper():
                    lines = section.strip().split('\n')
                    if len(lines) >= 2:
                        scene = lines[0].strip()
                        if ":" in scene:
                            scene = scene.split(":", 1)[1].strip()
                        prompt = " ".join(lines[1:]).strip()
                        if prompt and len(prompt) > 10:
                            prompt_data.append({"scene": scene, "prompt": prompt})
        
        # Hala prompt bulunamazsa, görsel analizinden yeni promptlar oluştur
        if not prompt_data and image_analysis:
            logger.warning("Hiç prompt bulunamadı, tekrar deneniyor...")
            
            # Daha basit bir sistem talimatı kullanarak tekrar dene
            simplification_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional prompt engineer. Create 4 separate product photography prompts based on the analysis below. Each prompt should be complete, detailed and ready to use with AI image generation. Format your response with SCENE1, SCENE2, etc. followed by the prompt text."},
                    {"role": "user", "content": f"Product Analysis:\n{image_analysis}\n\nCreate 4 product photography prompts."}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            simple_response = simplification_response.choices[0].message.content.strip()
            logger.info(f"Basitleştirilmiş prompt yanıtı: {simple_response[:100]}...")
            
            # Bu yanıtı işle
            simple_sections = simple_response.split('\n\n')
            
            for section in simple_sections:
                lines = section.strip().split('\n')
                if not lines:
                    continue
                    
                scene_line = lines[0]
                if "SCENE" in scene_line.upper() and len(lines) > 1:
                    scene = ""
                    if ":" in scene_line:
                        scene = scene_line.split(":", 1)[1].strip()
                    else:
                        scene = scene_line.replace("SCENE", "").replace("Scene", "").strip()
                        
                    prompt = " ".join(lines[1:]).strip()
                    if prompt and len(prompt) > 10:
                        prompt_data.append({"scene": scene, "prompt": prompt})
        
        # Hiçbir şekilde prompt oluşturulamazsa, son çare olarak sabit promptlar ekle
        if not prompt_data:
            logger.warning("Hiçbir şekilde prompt oluşturulamadı, sabit promptlar kullanılıyor")
            prompt_data = [
                {
                    "scene": "Professional Studio",
                    "prompt": "A professional product photo with bright studio lighting, clean white background, high contrast, sharp details, professional product photography, commercial-grade, marketing material, advertising image, perfect lighting, glossy finish, product showcase, detailed texture, professional photo shoot, pristine condition."
                },
                {
                    "scene": "Lifestyle Context",
                    "prompt": "Product in lifestyle context, natural environment, real-world usage, soft natural lighting, subtle background blur, photorealistic, high-resolution image, professional photography, authentic setting, practical demonstration, daily use scenario, relatable context, genuine interaction, vibrant colors."
                },
                {
                    "scene": "Dramatic Lighting",
                    "prompt": "Product with dramatic side lighting, high contrast, deep shadows, spotlight effect, dark background, cinematic look, professional product photography, artistic composition, moody atmosphere, premium appearance, elegant presentation, luxury product shot, emotional impact, striking visual."
                },
                {
                    "scene": "Technical Detail",
                    "prompt": "Close-up product details, macro photography, textural elements, fine details visible, technical precision, professional product documentation, informative image, clarity of design, engineering showcase, material quality highlight, craftsmanship focus, instructional photography, educational content, specification demonstration."
                }
            ]
        
        # Eğer 4'ten az prompt varsa, eksik olanları doldur
        while len(prompt_data) < 4 and len(prompt_data) > 0:
            prompt_data.append(prompt_data[0])  # İlk promptu tekrarla
        
        # Sadece ilk 4 promptu al
        prompt_data = prompt_data[:4]
        
        logger.info(f"Oluşturulan prompt sayısı: {len(prompt_data)}")
        for idx, p in enumerate(prompt_data):
            logger.info(f"Prompt {idx+1} - {p['scene']}: {p['prompt'][:50]}...")
        
        # Sonucu döndür
        return {
            "input_text": text,
            "feature_type": feature_type,
            "source_type": source_type,
            "prompt_data": prompt_data
        }
        
    except Exception as e:
        logger.error(f"Prompt oluşturulurken hata: {str(e)}")
        logger.error(f"Hata izleme: {traceback.format_exc()}")
        raise ValueError(f"Prompt oluşturulurken hata: {str(e)}")

# Initialize Supabase client
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

logger.info(f"SUPABASE_URL present: {bool(SUPABASE_URL)}")
logger.info(f"SUPABASE_KEY present: {bool(SUPABASE_KEY)}")

supabase = None
try:
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Supabase credentials not found in environment variables")
        logger.error("Please make sure SUPABASE_URL and SUPABASE_KEY are set")
    else:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully")
        
        # Test connection by querying the database
        logger.info("Testing Supabase connection...")
        test_response = supabase.table('generations').select("*").limit(1).execute()
        logger.info(f"Supabase connection test successful! Found {len(test_response.data if test_response.data else [])} records")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {str(e)}")
    logger.error(f"Error type: {type(e).__name__}")
    logger.error(f"Error details: {traceback.format_exc()}")
    supabase = None

def fetch_generations_from_db(limit=100, offset=0):
    """Fetch generations from Supabase database."""
    try:
        logger.info(f"Fetching generations from database (limit: {limit}, offset: {offset})")
        
        if not supabase:
            logger.error("Supabase client not initialized")
            return []
        
        logger.info("Building Supabase query...")
        query = supabase.table('generations').select("*").order('created_at', desc=True).limit(limit)
        
        # Check if the offset method is available (depends on Supabase version)
        try:
            logger.info("Trying to use offset method...")
            has_offset = hasattr(query, 'offset')
            
            if has_offset:
                logger.info(f"Offset method available, using offset {offset}...")
                response = query.offset(offset).execute()
            else:
                logger.info("Offset method not available, using alternative pagination...")
                # For older versions, get more records and slice manually
                larger_response = query.execute()
                
                # Get data and manually apply offset
                data = larger_response.data if larger_response.data else []
                if len(data) > offset:
                    # Slice the data to simulate offset+limit
                    larger_response.data = data[offset:offset+limit]
                else:
                    larger_response.data = []
                    
                response = larger_response
        except AttributeError as e:
            logger.warning(f"Offset method error: {str(e)}")
            logger.info("Falling back to basic query without offset...")
            response = query.execute()
            
            # Get data and manually apply offset
            data = response.data if response.data else []
            if len(data) > offset:
                # Slice the data to simulate offset+limit
                response.data = data[offset:offset+limit]
            else:
                response.data = []
        
        logger.info(f"Received response from Supabase with {len(response.data if response.data else [])} records")
        
        # Process the data to ensure all required fields are present
        processed_data = []
        for item in response.data:
            processed_item = {
                'id': item.get('id'),
                'url': item.get('url'),
                'type': item.get('type', ''),  # Media type (image/video)
                'content_type': item.get('content_type', ''),  # Section type (creative-scene/product-visual/video-image)
                'prompt': item.get('prompt', 'No prompt provided'),
                'model': item.get('model', ''),  # Model used for generation
                'duration': item.get('duration', 0),  # Duration (for videos)
                'created_at': item.get('created_at')
            }
            
            # Format the date if present
            if processed_item['created_at']:
                try:
                    if isinstance(processed_item['created_at'], str):
                        created_at = datetime.fromisoformat(processed_item['created_at'].replace('Z', '+00:00'))
                    else:
                        created_at = processed_item['created_at']
                    processed_item['date'] = created_at.strftime("%Y-%m-%d %H:%M:%S")
                except Exception as date_error:
                    logger.error(f"Error formatting date: {str(date_error)}")
                    processed_item['date'] = "Unknown date"
            
            processed_data.append(processed_item)
        
        logger.info(f"Processed {len(processed_data)} records from database")
        return processed_data
    except Exception as e:
        logger.error(f"Error fetching generations from database: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return []

def store_generated_content(url, content_type, type, prompt=None, model=None, duration=0):
    """Store content in the Supabase database and return the record."""
    try:
        # Validate the input
        if not url:
            logger.error("Cannot store content: URL is required")
            raise ValueError("URL is required")
        
        if not type:
            logger.error("Cannot store content: Type is required")
            raise ValueError("Type is required")
        
        valid_types = ['image', 'video','audio']
        if type not in valid_types:
            logger.error(f"Invalid type: {type}. Must be one of: {', '.join(valid_types)}")
            raise ValueError(f"Invalid type. Must be one of: {', '.join(valid_types)}")

        # Check if Supabase client is initialized
        if not supabase:
            logger.error("Supabase client is not initialized. Cannot store content.")
            return None

        # Ensure duration is an integer value
        try:
            # If it's already an integer, use it directly
            if isinstance(duration, int):
                duration_value = duration
            # Otherwise extract digits and convert to integer
            else:
                duration_digits = ''.join(filter(str.isdigit, str(duration)))
                duration_value = int(duration_digits) if duration_digits else 0
        except Exception as e:
            logger.error(f"Error converting duration to integer: {str(e)}. Using default value 0.")
            duration_value = 0

        # Log incoming data
        logger.info(f"Storing content: url={url}, content_type={content_type}, type={type}, prompt={prompt[:50] if prompt else None}, model={model}, duration={duration_value}...")

        # Get current credit value before insertion
        current_credit = None
        try:
            credit_response = supabase.table('generations') \
                .select('credit') \
                .order('created_at', desc=True) \
                .limit(1) \
                .execute()
            
            if credit_response and credit_response.data and len(credit_response.data) > 0:
                current_credit = credit_response.data[0].get('credit', 100)
            else:
                current_credit = 100  # Default if no records exist
            
            logger.info(f"Current credit before insertion: {current_credit}")
        except Exception as e:
            logger.error(f"Error getting current credit: {str(e)}")
            current_credit = 100  # Default if error

        # Prepare data for insertion - trigger will handle credit reduction
        data = {
            'url': url,
            'content_type': content_type,
            'type': type,
            'prompt': prompt,
            'model': model,
            'duration': duration_value,
            'credit': current_credit,  # Set current credit, let the trigger handle reduction
            'created_at': datetime.now().isoformat()
        }

        logger.info(f"Prepared data for insertion: {data}")

        # Insert into database
        logger.info("Sending insert request to Supabase...")
        response = supabase.table('generations').insert(data).execute()
        
        logger.info(f"Insert response from Supabase: {response}")
        
        if hasattr(response, 'error') and response.error is not None:
            logger.error(f"Supabase error during insertion: {response.error}")
            raise Exception(f"Database error: {response.error}")

        logger.info(f"Successfully stored content in database. Response data: {response.data[0] if response.data else None}")
        return response.data[0] if response.data else None

    except Exception as e:
        logger.error(f"Error storing content: {str(e)}")
        logger.error(traceback.format_exc())
        # Don't raise the exception, just log it and return None
        # This way, even if storing fails, the rest of the application can continue
        return None

# User authentication function
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Authentication routes
@app.route('/auth')
def auth():
    """Redirect legacy auth route to the login page"""
    # Check if coming from intro page by checking the referrer
    referrer = request.referrer
    if referrer and ('/intro' in referrer):
        # If coming from intro, proceed to login
        return redirect(url_for('login'))
    else:
        # If directly accessing /auth, go to intro first
        return redirect(url_for('intro'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle login form submission"""
    # If direct access (not from intro) and it's a GET request, redirect to intro
    referrer = request.referrer
    from_intro = request.args.get('from') == 'intro'
    
    if request.method == 'GET' and not (referrer and ('/intro' in referrer)) and not from_intro:
        # Check if there's a session cookie already set (meaning we're already logged in)
        if session.get('logged_in'):
            logger.info("User already logged in, redirecting to welcome page")
            return redirect(url_for('welcome_direct'))
        else:
            # If not logged in and not coming from intro, go to intro first
            logger.info("Direct access to login page, redirecting to intro first")
            return redirect(url_for('intro'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember_me = True if request.form.get('remember-me') else False
        
        logger.info(f"Login attempt for email: {email}")
        
        # Load and validate credentials from static auth_config.json
        auth_config_path = os.path.join(app.root_path, 'static', 'auth_config.json')
        try:
            with open(auth_config_path) as cfg_file:
                valid_creds = json.load(cfg_file).get('validCredentials', {})
        except Exception as e:
            logger.error(f"Error loading auth config: {e}")
            valid_creds = {}
        if email == valid_creds.get('email') and password == valid_creds.get('password'):
            # Extract name from email (everything before @)
            display_name = email.split('@')[0].title()
            
            # Set session variables
            session['user_id'] = str(uuid.uuid4())  # Generate a random user ID
            session['email'] = email
            session['display_name'] = display_name
            session['logged_in'] = True
            session['auth_completed'] = True  # Flag to indicate auth is completed and should bypass intro
            session.permanent = remember_me  # Set session to permanent if remember me is checked
            
            logger.info(f"Login successful for email: {email}")
            return redirect(url_for('welcome_direct'))
        else:
            logger.warning(f"Login failed for email: {email}")
            # In a real app, you would use flash messages to show errors
            return render_template('auth.html', error="Invalid email or password")
    
    # GET request - show the login form
    return render_template('auth.html')

@app.route('/logout')
def logout():
    """Handle user logout"""
    # Clear session data
    session.pop('user_id', None)
    session.pop('email', None)
    session.pop('display_name', None)
    session.pop('logged_in', None)
    
    # Optional: clear the session completely
    session.clear()
    
    logger.info("User logged out")
    return redirect(url_for('login'))

# Modified main route to check for authentication
@app.route('/')
def welcome():
    logger.info("Landing page accessed")
    # Clear any existing localStorage to ensure a fresh start
    logger.info("Redirecting to intro page")
    return redirect(url_for('intro'))

@app.route('/intro')
def intro():
    """Show intro video without requiring login"""
    logger.info("Intro page displayed")
    return render_template('intro.html')

@app.route('/welcome')
def welcome_direct():
    """Doğrudan karşılama sayfasını göster"""
    # Check if user is logged in
    if not session.get('logged_in'):
        logger.info("User not logged in, redirecting to login page")
        return redirect(url_for('login'))
    
    # Get the user's display name from session
    display_name = session.get('display_name', 'User')
    
    # Get source parameter if it exists
    source = request.args.get('source', '')
    
    # Check if coming from auth completion
    auth_completed = session.get('auth_completed', False)
    if auth_completed:
        # Reset the flag for future navigation
        session['auth_completed'] = False
        logger.info("Auth completed, rendering welcome page directly")
    
    logger.info("Karşılama sayfası doğrudan görüntüleniyor")
    return render_template('welcome.html', display_name=display_name, source=source, auth_completed=auth_completed)

@app.route('/home')
def home_direct():
    """Navigation bar Home link handler - bypasses intro page"""
    # Check if user is logged in
    if not session.get('logged_in'):
        logger.info("User not logged in, redirecting to login page")
        return redirect(url_for('login'))
    
    logger.info("Home link clicked, bypassing intro page")
    return redirect(url_for('welcome_direct', source='home'))

@app.route('/index')
def index():
    """Ana uygulama sayfasını göster"""
    logger.info("Ana uygulama sayfası görüntüleniyor")
    return render_template('index.html')

@app.route('/image')
def image():
    """Görsel üretici sayfasını göster"""
    image_urls = request.args.getlist('image_url')  # Birden fazla görsel URL'si alabilmek için getlist kullan
    prompt = request.args.get('prompt')
    brand = request.args.get('brand')
    
    if not image_urls:
        logger.info("Görsel üretici sayfası görüntüleniyor")
        return render_template('image.html')
    
    logger.info(f"Görsel sonuç sayfası görüntüleniyor. Görsel URL sayısı: {len(image_urls)}")
    return render_template('image.html', image_urls=image_urls, prompt=prompt, brand=brand)

@app.route("/generate-prompt", methods=["POST"])
def generate_prompt_api():
    """API endpoint for generating prompts."""
    data = request.json
    text = data.get("text")
    feature_type = data.get("feature_type")
    source_type = data.get("source_type", "url")  # Default to 'url' if not provided
    
    if not text or not feature_type:
        logger.error("Missing required parameters in generate_prompt_api")
        return jsonify({"error": "Missing required parameters: 'text' and 'feature_type'"}), 400
    
    logger.info(f"Generating prompt for text: {text[:100]}... Feature type: {feature_type}, Source type: {source_type}")
    
    try:
        result = generate_prompt(text, feature_type, source_type)
        logger.info(f"Successfully generated prompts: {json.dumps(result)[:200]}...")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in generate_prompt_api: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_video', methods=['POST'])
def generate_video():
    prompt = request.form.get('prompt')
    brand_input = request.form.get('brand_input')
    aspect_ratio = request.form.get('aspect_ratio', '9:16')  # Varsayılan olarak 9:16
    duration = request.form.get('duration', '8s')  # Yeni: frontend'den süre değerini al
    content_type = request.form.get('content_type', 'creative-scene')  # Default to creative-scene for index.html/video.html
    
    if not prompt:
        return jsonify({"error": "Geçersiz prompt seçimi"}), 400
    
    # Fal.ai client'ın kullanılabilir olup olmadığını kontrol et
    if not FAL_CLIENT_AVAILABLE:
        logger.error("fal_client kütüphanesi yüklü değil. Video oluşturulamıyor.")
        return jsonify({"error": "Video oluşturma özelliği şu anda kullanılamıyor. Sunucu yapılandırması eksik."}), 500
    
    try:
        logger.info(f"Fal.ai API'sine video oluşturma isteği gönderiliyor")
        logger.info(f"Kullanılan prompt: {prompt[:50]}...")  # İlk 50 karakteri logla
        logger.info(f"Kullanılan aspect ratio: {aspect_ratio}")
        logger.info(f"Kullanılan süre: {duration}")
        logger.info(f"Content type: {content_type}")
        
        # Fal.ai Veo2 API'si ile video oluştur
        try:
            logger.info("Fal.ai istemcisi ile video oluşturuluyor...")
            
            # Benzersiz bir istek ID'si oluştur (sadece loglama için)
            request_id = str(uuid.uuid4())
            logger.info(f"Oluşturulan istek ID: {request_id}")
            
            # İlerleme güncellemelerini işlemek için callback fonksiyonu
            def on_queue_update(update):
                if hasattr(update, 'logs') and update.logs:
                    for log in update.logs:
                        logger.info(f"🔄 {log.get('message', '')}")
                
                if hasattr(update, 'status'):
                    logger.info(f"Fal.ai durum: {update.status}")
            
            # API isteği için parametreler
            arguments = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,  # Kullanıcının seçtiği aspect ratio
                "duration": duration  # Kullanıcının seçtiği süre
            }
            
            # Parametreleri logla
            logger.info(f"Fal.ai parametreleri: {json.dumps(arguments)}")
            
            # İstek zamanını ölç
            request_start_time = time.time()
            logger.info("Fal.ai isteği başlıyor...")
            
            # Fal.ai Veo2 modelini çağır
            result = fal_client.subscribe(
                "fal-ai/veo2",
                arguments=arguments,
                with_logs=True,
                on_queue_update=on_queue_update
            )
            
            request_duration = time.time() - request_start_time
            logger.info(f"Fal.ai isteği tamamlandı. Süre: {request_duration:.2f} saniye")
            
            # Sonucu logla
            logger.info(f"Fal.ai sonucu: {json.dumps(result)[:200]}...")  # İlk 200 karakteri logla
            
            # Video URL'sini al
            logger.info("Video URL'si alınıyor...")
            video_url = result.get("video", {}).get("url")
            
            if not video_url:
                logger.error(f"Video URL'si bulunamadı. Sonuç: {result}")
                return jsonify({"error": "Video URL'si alınamadı"}), 500
            
            logger.info(f"Video başarıyla oluşturuldu. URL: {video_url}")
            
            # Store the generated video in our library
            store_generated_content(
                url=video_url,
                content_type=content_type,  # Use the content_type from request
                type="video",
                prompt=prompt,
                model="fal-ai/veo2",
                duration=duration  # Pass raw duration value
            )
            
            # Video sayfasına yönlendir
            logger.info("İstemciye yanıt gönderiliyor...")
            return jsonify({
                "video_url": video_url,
                "prompt": prompt
            })
            
        except Exception as fal_error:
            logger.error(f"Fal.ai istemcisi hatası: {str(fal_error)}")
            logger.error(f"Hata türü: {type(fal_error).__name__}")
            logger.error(f"Hata detayları: {str(fal_error)}")
            logger.error(f"Hata izleme: {traceback.format_exc()}")
            
            # Alternatif olarak REST API'yi dene
            logger.info("Fal.ai istemcisi başarısız oldu, REST API deneniyor...")
            try:
                # API isteği için başlıklar
                headers = {
                    "Authorization": f"Key {FAL_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                if duration and ''.join(filter(str.isdigit, str(duration))).isdigit():
                    digits = ''.join(filter(str.isdigit, str(duration)))
                    suffix = ''.join(filter(lambda x: not x.isdigit(), str(duration)))
                    duration_param = f"{digits}{suffix if suffix else 's'}"
                else:
                    duration_param = "8s"  # Default if no valid duration
                
                # API isteği için veri
                payload = {
                    "input": {
                        "prompt": prompt,
                        "aspect_ratio": aspect_ratio,  # Kullanıcının seçtiği aspect ratio
                        "duration": duration_param  # Format the duration appropriately
                    }
                }
                
                # API isteği gönder
                logger.info("REST API isteği gönderiliyor...")
                response = requests.post(
                    "https://api.fal.ai/v1/video/veo2",
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                
                # Yanıtı kontrol et
                if response.status_code != 200:
                    logger.error(f"REST API hatası: {response.text}")
                    return jsonify({"error": f"Video oluşturma başarısız oldu: {response.text}"}), 500
                
                # Yanıtı JSON olarak ayrıştır
                result = response.json()
                
                # Video URL'sini al
                video_url = result.get("video", {}).get("url")
                
                if not video_url:
                    logger.error(f"Video URL'si bulunamadı. Sonuç: {result}")
                    return jsonify({"error": "Video URL'si alınamadı"}), 500
                
                logger.info(f"REST API ile video başarıyla oluşturuldu. URL: {video_url}")
                
                # Store the generated video in our library
                store_generated_content(
                    url=video_url,
                    content_type=content_type,  # Use the content_type from request
                    type="video",
                    prompt=prompt,
                    model="fal-ai/veo2",
                    duration=duration  # Pass raw duration value
                )
                
                # Video sayfasına yönlendir
                return jsonify({
                    "video_url": video_url,
                    "prompt": prompt
                })
                
            except Exception as rest_error:
                logger.error(f"REST API hatası: {str(rest_error)}")
                logger.error(f"Hata izleme: {traceback.format_exc()}")
                return jsonify({"error": f"Video oluşturma başarısız oldu: {str(rest_error)}"}), 500
            
            return jsonify({"error": f"Video oluşturma başarısız oldu: {str(fal_error)}"}), 500
    
    except Exception as e:
        error_msg = f"Hata: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Hata izleme: {traceback.format_exc()}")
        
        return jsonify({"error": f"Bir hata oluştu: {str(e)}"}), 500

@app.route('/video')
def video():
    video_url = request.args.get('video_url')
    prompt = request.args.get('prompt')
    brand = request.args.get('brand')
    
    if not video_url:
        return redirect(url_for('index'))
    
    logger.info(f"Video sayfası görüntüleniyor. Video URL: {video_url}")
    return render_template('video.html', video_url=video_url, prompt=prompt, brand=brand)

@app.route('/check_status/<request_id>')
def check_status(request_id):
    """Check the status of an image generation request."""
    try:
        if not FAL_CLIENT_AVAILABLE:
            raise Exception("fal.ai client kütüphanesi yüklü değil")

        logger.info(f"Checking request status (ID: {request_id})...")
        
        # Initialize fal.ai client
        fal_client.api_key = os.getenv('FAL_KEY')
        
        # Get status from fal.ai
        status = fal_client.get_queue_status(request_id)
        
        # Convert status object to dictionary
        status_dict = {
            "status": "completed" if status.get('completed', False) else "in_progress",
            "logs": status.get('logs', [])
        }
        
        return jsonify(status_dict)
            
    except Exception as e:
        logger.error(f"Error checking request status: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "logs": []
        }), 500

@app.route('/debug')
def debug():
    """Debug endpoint to check environment variables and configuration"""
    debug_info = {
        "python_version": sys.version,
        "environment": os.environ.get('VERCEL_ENV', 'local'),
        "openai_api_key_exists": bool(OPENAI_API_KEY),
        "fal_api_key_exists": bool(FAL_API_KEY),
        "astria_api_key_exists": bool(ASTRIA_API_KEY),
        "template_dir_exists": os.path.exists(template_dir),
        "templates": [f for f in os.listdir(template_dir) if os.path.isfile(os.path.join(template_dir, f))] if os.path.exists(template_dir) else []
    }
    return jsonify(debug_info)

@app.route('/debug/env')
def debug_env():
    """Debug endpoint to check environment variables"""
    # Create a safe version of environment variables (only show presence, not values)
    env_info = {
        "SUPABASE_URL_present": bool(os.getenv("SUPABASE_URL")),
        "SUPABASE_KEY_present": bool(os.getenv("SUPABASE_KEY")),
        "FAL_API_KEY_present": bool(os.getenv("FAL_API_KEY")),
        "FAL_KEY_present": bool(os.getenv("FAL_KEY")),
        "OPENAI_API_KEY_present": bool(os.getenv("OPENAI_API_KEY")),
        "FLASK_ENV": os.getenv("FLASK_ENV"),
        "PORT": os.getenv("PORT"),
        "supabase_initialized": supabase is not None,
        "FAL_CLIENT_AVAILABLE": FAL_CLIENT_AVAILABLE
    }
    
    # Also check if env variables have correct format
    if os.getenv("SUPABASE_URL"):
        env_info["SUPABASE_URL_format_valid"] = os.getenv("SUPABASE_URL").startswith("https://") and ".supabase.co" in os.getenv("SUPABASE_URL")
    
    if os.getenv("SUPABASE_KEY"):
        key = os.getenv("SUPABASE_KEY")
        env_info["SUPABASE_KEY_format_valid"] = len(key) > 50 and "." in key
    
    return jsonify(env_info)

# Initialize scrape.do configuration
SCRAPE_DO_API_KEY = os.getenv("SCRAPE_DO_API_KEY")
SCRAPE_DO_BASE_URL = "http://api.scrape.do"

@app.route('/extract-images', methods=['POST'])
def extract_images():
    try:
        # Get the URL from the request
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({"error": "URL is required"}), 400

        if not SCRAPE_DO_API_KEY:
            return jsonify({"error": "Scrape.do API key not configured"}), 500

        try:
            # Configuration for image extraction
            config = {
                "carousel_images": {
                    "selectors": [
                        "#pdp-carousel__slide0 img",
                        "#pdp-carousel__slide1 img",
                        "#pdp-carousel__slide2 img",
                        "#pdp-carousel__slide3 img",
                        "#pdp-carousel__slide4 img"
                    ],
                    "attribute": "src",
                    "filters": {
                        "include": "424-600/",
                        "endsWith": ".jpg",
                        "orInclude": "/format:webp"
                    }
                }
            }
            from urllib.parse import quote
            encoded_url = quote(url)
            max_retries = 3
            timeout_value = 120 
            
            for retry_count in range(max_retries):
                try:
                    logger.info(f"Making scrape.do request (attempt {retry_count + 1}/{max_retries}) with timeout {timeout_value}s")
                    scrape_url = f"{SCRAPE_DO_BASE_URL}?token={SCRAPE_DO_API_KEY}&url={encoded_url}"
                    response = requests.get(scrape_url, timeout=timeout_value)
                    
                    if response.status_code == 200:
                        logger.info(f"Scrape.do request successful on attempt {retry_count + 1}")
                        break
                    else:
                        logger.warning(f"Received status code {response.status_code} on attempt {retry_count + 1}")
                        if retry_count < max_retries - 1:
                            time.sleep(3) 
                
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout occurred on attempt {retry_count + 1}")
                    if retry_count < max_retries - 1:
                        time.sleep(3)
                        timeout_value += 30
                    else:
                        logger.error("All retry attempts failed with timeout")
                        raise
                
                except Exception as e:
                    logger.error(f"Error during request on attempt {retry_count + 1}: {str(e)}")
                    if retry_count < max_retries - 1:
                        time.sleep(3)
                    else:
                        raise
            
            if response.status_code != 200:
                return jsonify({"error": f"Failed to fetch URL after {max_retries} attempts: {response.status_code}"}), response.status_code
                
            # Parse the HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            unique_images = set()
            for selector in config["carousel_images"]["selectors"]:
                images = soup.select(selector)
                for img in images:
                    src = img.get(config["carousel_images"]["attribute"])
                    if src:
                        # Handle relative URLs
                        if src.startswith('//'):
                            src = 'https:' + src
                        elif src.startswith('/'):
                            # Get the base URL
                            from urllib.parse import urlparse
                            parsed_uri = urlparse(url)
                            base_url = f'{parsed_uri.scheme}://{parsed_uri.netloc}'
                            src = base_url + src
                        
                        filters = config["carousel_images"]["filters"]
                        if src.startswith('http') and not src.startswith('data:'):
                            if (filters["include"] in src and 
                                (src.endswith(filters["endsWith"]) or filters["orInclude"] in src)):
                                unique_images.add(src)
            
            # Convert set to list for JSON serialization
            extracted_images = list(unique_images)
            
            # Remove duplicate resolutions of the same image
            filtered_images = []
            base_urls = set()
            
            for img_url in extracted_images:
                base_img_url = img_url.split('?')[0]
                parts = base_img_url.split('/')
                if len(parts) > 1:
                    filename = parts[-1]
                    base_filename = re.sub(r'[-_]\d+[-_x]\d+', '', filename)
                    base_identifier = '/'.join(parts[:-1]) + '/' + base_filename
                else:
                    base_identifier = base_img_url
                
                if base_identifier not in base_urls:
                    base_urls.add(base_identifier)
                    filtered_images.append(img_url)
            
            logger.info(f"Found {len(filtered_images)} unique carousel images after filtering")
            for idx, img_url in enumerate(filtered_images, 1):
                logger.info(f"Image {idx}: {img_url}")
            
            return jsonify({
                "product_images": filtered_images
            })
            
        except Exception as e:
            logger.error(f"Error scraping URL: {str(e)}")
            return jsonify({"error": f"Failed to scrape URL: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Error in extract_images: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_image', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        prompt_data = data.get('prompt_data')
        image_url = data.get('image_url')
        content_type = data.get('content_type', 'product-visual')  # Default to product-visual if not specified

        if not prompt_data or not isinstance(prompt_data, list):
            return jsonify({"error": "Prompt verisi gerekli ve liste formatında olmalı"}), 400

        if content_type not in ['product-visual', 'video-image']:
            return jsonify({"error": "Invalid content type. Must be 'product-visual' or 'video-image'"}), 400

        if not FAL_CLIENT_AVAILABLE:
            return jsonify({"error": "fal.ai client kütüphanesi yüklü değil"}), 500

        logger.info(f"🎨 Çoklu görsel üretme isteği başlatılıyor...")
        logger.info(f"📝 Prompt sayısı: {len(prompt_data)}")
        logger.info(f"🔗 Referans görsel: {image_url}")
        
        # URL validation and preprocessing logic
        if image_url and isinstance(image_url, str):
            # Check if this is a product page URL rather than an image URL
            if not (image_url.lower().endswith('.jpg') or image_url.lower().endswith('.jpeg') or 
                    image_url.lower().endswith('.png') or image_url.lower().endswith('.webp') or
                    image_url.startswith('data:image/')):
                
                # Looks like a product page URL, log a warning
                logger.warning(f"⚠️ URL appears to be a product page, not a direct image URL: {image_url}")
                logger.warning("This URL might need to go through extract-images endpoint first")
                
                # Check if the URL contains hepsiburada or other e-commerce domains
                if 'hepsiburada.com' in image_url or 'trendyol.com' in image_url or 'amazon.com' in image_url:
                    logger.warning(f"⚠️ E-commerce product URL detected. These may not work directly with image generation API")
        
        # Fal.ai API anahtarını ayarla
        fal_client.api_key = os.getenv('FAL_KEY')

        generated_images = []

        def on_queue_update(update):
            if hasattr(update, 'logs') and update.logs:
                for log in update.logs:
                    logger.info(f"🔄 {log.get('message', '')}")

        # Her prompt için görsel üret
        for idx, prompt_item in enumerate(prompt_data, 1):
            prompt = prompt_item.get('prompt')
            scene = prompt_item.get('scene')
            
            logger.info(f"🎯 {idx}. görsel üretiliyor...")
            logger.info(f"📝 Sahne: {scene}")
            logger.info(f"📝 Prompt: {prompt}")

            try:
                # Log the full request to fal.ai for debugging
                request_params = {
                    "prompt": prompt,
                    "image_url": image_url,
                    "negative_prompt": "ugly, disfigured, low quality, blurry, nsfw",
                    "num_inference_steps": 28,
                    "guidance_scale": 5,
                    "seed": random.randint(1, 1000000)
                }
                logger.info(f"🔍 Sending request to fal.ai/iclight-v2 with params: {json.dumps(request_params)}")
                
                # Fal.ai'ye istek gönder
                result = fal_client.subscribe(
                    "fal-ai/iclight-v2",
                    arguments=request_params,
                    with_logs=True,
                    on_queue_update=on_queue_update
                )

                if not result or not isinstance(result, dict):
                    logger.error(f"❌ {idx}. görsel üretilemedi: Geçersiz yanıt formatı")
                    logger.error(f"Received result: {result}")
                    continue

                generated_image_url = result.get("images", [{}])[0].get("url")
                
                if not generated_image_url:
                    logger.error(f"❌ {idx}. görsel için URL bulunamadı")
                    logger.error(f"Full response from fal.ai: {json.dumps(result)}")
                    continue

                # Store the generated image in our library
                try:
                    store_generated_content(
                        url=generated_image_url,
                        content_type=content_type,
                        type="image",
                        prompt=prompt,
                        model="fal-ai/iclight-v2",
                    )
                    logger.info(f"✅ {idx}. görsel başarıyla veritabanına kaydedildi")
                except Exception as storage_error:
                    logger.error(f"❌ {idx}. görsel veritabanına kaydedilemedi: {str(storage_error)}")
                    # Continue with the process even if storage fails
                    # We'll still show the image to the user
                    pass

                generated_images.append({
                    "scene": scene,
                    "prompt": prompt,
                    "image_url": generated_image_url,
                    "request_id": result.get('request_id')
                })

                logger.info(f"✨ {idx}. görsel başarıyla üretildi")
                logger.info(f"🖼️ Üretilen görsel URL: {generated_image_url}")

            except Exception as e:
                logger.error(f"❌ {idx}. görsel üretilirken hata: {str(e)}")
                logger.error(f"Hata izleme: {traceback.format_exc()}")
                continue

        if not generated_images:
            logger.error("❌ Hiçbir görsel üretilemedi. Bildirilen hataları kontrol edin.")
            
            # Give a more specific error message for common URL problems
            if image_url and 'hepsiburada.com' in image_url and not image_url.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                return jsonify({
                    "error": "Hiçbir görsel üretilemedi. URL bir ürün sayfasıymış gibi görünüyor, görsel URL'si değil. Lütfen önce URL'den görselleri çıkarıp, bir görsel URL'si seçin."
                }), 400
            
            return jsonify({"error": "Hiçbir görsel üretilemedi"}), 500

        return jsonify({
            "status": "success",
            "generated_images": generated_images
        })

    except Exception as e:
        logger.error(f"❌ Görsel üretme hatası: {str(e)}")
        logger.error(f"Hata izleme: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/image2')
def image2():
    """Görsel üretici sayfasını göster"""
    image_urls = request.args.getlist('image_url')  # Birden fazla görsel URL'si alabilmek için getlist kullan
    prompt = request.args.get('prompt')
    brand = request.args.get('brand')
    prompt_id = request.args.get('prompt_id')
    
    # Eğer prompt_id varsa ve görsel URL'leri yoksa, check_image_status fonksiyonunu çağır
    if prompt_id and not image_urls:
        try:
            # API bilgilerini al
            api_key = os.getenv("ASTRIA_API_KEY")
            
            if not api_key:
                logger.error("API yapılandırması eksik")
                return render_template('image2.html')
            
            # Flux model ID - Astria'nın genel Flux modelini kullanıyoruz
            flux_model_id = "1504944"  # Flux1.dev from the gallery
            
            # API URL'sini oluştur - prompt_id ile durumu kontrol et
            api_url = f"https://api.astria.ai/tunes/{flux_model_id}/prompts/{prompt_id}"
            
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            
            # API'ye istek gönder
            logger.info(f"Astria API durum kontrolü: {api_url}")
            response = requests.get(
                api_url,
                headers=headers
            )
            
            # Yanıtı kontrol et
            if response.status_code == 200:
                # Yanıtı JSON olarak parse et
                result = response.json()
                
                # Görsel URL'lerini farklı formatlarda kontrol et
                if 'images' in result and isinstance(result['images'], list) and len(result['images']) > 0:
                    for image in result['images']:
                        if isinstance(image, dict) and 'url' in image:
                            image_urls.append(image.get('url'))
                        elif isinstance(image, str):
                            image_urls.append(image)
                
                # Diğer olası formatları kontrol et
                if not image_urls and 'image_url' in result:
                    image_urls.append(result.get('image_url'))
                if not image_urls and 'output' in result and isinstance(result['output'], dict) and 'image_url' in result['output']:
                    image_urls.append(result['output']['image_url'])
                
                # Görsel URL'lerini loglama
                if image_urls:
                    logger.info(f"Toplam {len(image_urls)} görsel URL bulundu")
                    logger.info(f"İlk görsel URL: {image_urls[0]}")
        except Exception as e:
            logger.error(f"Görsel durumu kontrol edilirken hata oluştu: {str(e)}")
    
    # Tek bir URL string olarak geldiyse, onu listeye çevir
    if not image_urls and request.args.get('image_url'):
        image_urls = [request.args.get('image_url')]
    
    if not image_urls:
        logger.info("Görsel üretici sayfası görüntüleniyor")
        return render_template('image2.html')
    
    logger.info(f"Görsel sonuç sayfası görüntüleniyor. Görsel URL sayısı: {len(image_urls)}")
    return render_template('image2.html', image_urls=image_urls, prompt=prompt, brand=brand, prompt_id=prompt_id)

@app.route("/generate-prompt-2", methods=["POST"])
def generate_prompt_2_api():
    """API endpoint for generating prompts."""
    data = request.json
    text = data.get("text")
    feature_type = data.get("feature_type")
    aspect_ratio = data.get("aspect_ratio", "1:1")  # Varsayılan olarak 1:1
    
    if not text or not feature_type:
        return jsonify({"error": "Missing required parameters: 'text' and 'feature_type'"}), 400
    
    try:
        result = generate_prompt_2(text, feature_type, aspect_ratio)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

def detect_style(text: str, feature_type: str) -> str:
    """
    OpenAI'ye ayrı bir istek atarak, girilen metne ve feature_type değerine göre promptun kendi stiline uygun bir stil belirler.
    """
    if feature_type == "image":
        instructions = """Objective:
        Analyze the given text and determine the most appropriate artistic style for an image based on its descriptive elements.
        The detected style should reflect the atmosphere, mood, and composition implied by the text.
        
        Consider factors such as:
        - Lighting (soft, dramatic, neon, natural, etc.)
        - Depth and perspective (wide-angle, close-up, aerial view, etc.)
        - Color palette (vibrant, monochrome, pastel, etc.)
        - Texture and rendering (hyperrealistic, sketch, painterly, etc.)
        
        Output Format:
        Provide a single style descriptor that encapsulates the detected artistic characteristics. Keep it concise and relevant to the provided text."""
    
    elif feature_type == "video":
        instructions = """Objective:
        Analyze the given text and determine the most appropriate cinematic style for a video based on its descriptive elements.
        The detected style should reflect the motion, pacing, and atmosphere implied by the text.
        
        Consider factors such as:
        - Camera movement (steady, shaky cam, sweeping drone shots, etc.)
        - Editing style (fast cuts, slow motion, time-lapse, etc.)
        - Lighting and mood (high contrast, natural, moody, vibrant, etc.)
        - Color grading (warm, cool, desaturated, high-contrast, etc.)
        
        Output Format:
        Provide a single style descriptor that encapsulates the detected cinematic characteristics. Keep it concise and relevant to the provided text."""
    
    else:
        raise ValueError("Geçersiz feature_type! 'image' veya 'video' olmalıdır.")
    
    logger.info(f"Stil belirleme isteği gönderiliyor. Metin: {text[:50]}... Özellik tipi: {feature_type}")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": f"Text: {text}\nFeature Type: {feature_type}\nDetermine the best style:"}
            ]
        )
        
        style = response.choices[0].message.content.strip()
        logger.info(f"Belirlenen stil: {style}")
        return style
    except Exception as e:
        logger.error(f"Stil belirlenirken hata: {str(e)}")
        logger.error(f"Hata izleme: {traceback.format_exc()}")
        raise ValueError(f"Stil belirlenirken hata: {str(e)}")

def generate_prompt_2(text: str, feature_type: str, aspect_ratio: str = "1:1") -> dict:
    """
    OpenAI chat completion API kullanarak doğrudan prompt oluşturur.
    Her bir prompt için ayrı stil belirler.
    """
    if feature_type not in ["image", "video"]:
        raise ValueError("Geçersiz feature_type! 'image' veya 'video' olmalıdır.")
    
    logger.info(f"Prompt oluşturuluyor. Metin: {text[:50]}... Özellik tipi: {feature_type}, Aspect Ratio: {aspect_ratio}")
    
    try:
        # Feature type değerini uygun formata dönüştür
        prompt_type = "image" if feature_type == "image" else "video"
        
        # Aspect ratio açıklaması
        aspect_ratio_desc = ""
        if aspect_ratio == "1:1":
            aspect_ratio_desc = "square format (1:1)"
        elif aspect_ratio == "4:5":
            aspect_ratio_desc = "portrait format for Instagram posts (4:5)"
        elif aspect_ratio == "16:9":
            aspect_ratio_desc = "landscape format for web/video (16:9)"
        elif aspect_ratio == "9:16":
            aspect_ratio_desc = "vertical format for stories/reels (9:16)"
        
        # Sistem talimatı - Her prompt için ayrı stil belirle
        system_instruction = f"""
        Görevin, kullanıcının verdiği metin için {prompt_type} oluşturmak üzere 4 farklı prompt üretmektir.  

                Her prompt için farklı bir yaratıcı yaklaşım ve stil belirle ve her promptun başına stilini ekle.  

                ### Kurallar:  
                1. Her prompt en az 50, en fazla 120 kelime olmalıdır. Daha kapsamlı ve detaylı açıklamalar için yeterli uzunluk sağlanmalıdır.  
                2. Her prompt farklı bir görsel ve anlatım yaklaşımı sunmalıdır. Stil, kompozisyon, atmosfer veya teknik bakış açılarıyla çeşitlilik yaratılmalıdır.  
                3. Promptlar doğrudan {prompt_type} oluşturmak için optimize edilmelidir. Her biri, ilgili modelin en iyi sonuçları vermesi için açık, detaylı ve yönlendirici olmalıdır.  
                4. Promptlar mutlaka İngilizce olmalıdır. Teknik ve yaratıcı detayların daha iyi işlenmesi için tüm açıklamalar İngilizce verilmelidir.  
                5. Promptlar {aspect_ratio_desc} için optimize edilmelidir.** Belirtilen en-boy oranına uygun çerçeveleme ve perspektif detayları içermelidir.  
                6. Görseller için ışık, renk paleti, perspektif ve detay seviyesi tanımlanmalıdır. Promptlar, modelin görsel uyumu sağlaması için estetik ve teknik öğeler içermelidir.  
                7. Videolar için hareket, tempo, kamera açısı ve stil detayları belirtilmelidir. Video içeriklerinde sahne akışı, kamera dinamikleri ve atmosfer önemlidir.  
                8. Her prompt, AI modelleri tarafından kolayca anlaşılabilir ve doğru yorumlanabilir olmalıdır. Fazla soyut veya muğlak ifadeler yerine, açık ve yönlendirici dil kullanılmalıdır.  

                ### Yanıt formatı:  

                STYLE1: [Birinci promptun stili]  
                [Prompt 1]  

                STYLE2: [İkinci promptun stili]  
                [Prompt 2]  

                STYLE3: [Üçüncü promptun stili]  
                [Prompt 3]  

                STYLE4: [Dördüncü promptun stili]  
                [Prompt 4]  
        """
        
        # Chat completion isteği gönder
        logger.info("Chat completion isteği gönderiliyor...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Metin: {text}\nTür: {feature_type}\nAspect Ratio: {aspect_ratio}"}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        
        # Yanıtı işle
        response_text = response.choices[0].message.content.strip()
        logger.info(f"GPT yanıtı alındı: {response_text[:100]}...")
        
        # Stil ve promptları ayır
        sections = response_text.split('\n\n')
        prompt_data = []
        
        for section in sections:
            lines = section.strip().split('\n')
            if not lines:
                continue
                
            # İlk satırdan stili çıkar
            style_line = lines[0]
            if "STYLE" in style_line.upper() and ":" in style_line:
                style = style_line.split(":", 1)[1].strip()
                # Stil satırını çıkar ve kalan satırları prompt olarak birleştir
                prompt = " ".join(lines[1:]).strip()
                if prompt and len(prompt) > 10:
                    prompt_data.append({"style": style, "prompt": prompt})
        
        # Eğer hiç prompt bulunamadıysa, metni doğrudan kullan
        if not prompt_data:
            logger.warning("Hiç prompt bulunamadı, metni doğrudan kullanıyoruz")
            prompt_data.append({
                "style": "default",
                "prompt": f"{text} {aspect_ratio} aspect ratio"
            })
        
        # Eğer 4'ten az prompt varsa, eksik olanları doldur
        while len(prompt_data) < 4 and len(prompt_data) > 0:
            prompt_data.append(prompt_data[0])  # İlk promptu tekrarla
        
        # Sadece ilk 4 promptu al
        prompt_data = prompt_data[:4]
        
        logger.info(f"Oluşturulan prompt sayısı: {len(prompt_data)}")
        
        # Sonucu döndür
        return {
            "input_text": text,
            "feature_type": feature_type,
            "aspect_ratio": aspect_ratio,
            "prompt_data": prompt_data
        }
        
    except Exception as e:
        logger.error(f"Prompt oluşturulurken hata: {str(e)}")
        logger.error(f"Hata izleme: {traceback.format_exc()}")
        raise ValueError(f"Prompt oluşturulurken hata: {str(e)}")

@app.route('/check_image_status/<prompt_id>', methods=['GET'])
def check_image_status(prompt_id):
    """Asenkron görsel oluşturma işleminin durumunu kontrol etmek için kullanılan endpoint"""
    try:
        # Yönlendirme seçeneğini kontrol et
        redirect_to_page = request.args.get('redirect', 'false').lower() == 'true'
        prompt = request.args.get('prompt', '')
        brand = request.args.get('brand', '')
        aspect_ratio = request.args.get('aspect_ratio', '1:1')  # Aspect ratio bilgisini al
    #    content_type = request.args.get('content_type', 'video-image')  # Changed default to video-image
        
        # API bilgilerini al
        api_key = os.getenv("ASTRIA_API_KEY")
        
        if not api_key:
            return jsonify({"error": "API yapılandırması eksik"}), 500
        
        # Flux model ID - Astria'nın genel Flux modelini kullanıyoruz
        flux_model_id = "1504944"  # Flux1.dev from the gallery
        
        # API URL'sini oluştur - prompt_id ile durumu kontrol et
        api_url = f"https://api.astria.ai/tunes/{flux_model_id}/prompts/{prompt_id}"
        
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        # API'ye istek gönder
        logger.info(f"Astria API durum kontrolü: {api_url}")
        response = requests.get(
            api_url,
            headers=headers
        )
        
        # Yanıtı kontrol et
        if response.status_code == 200:
            # Yanıtı JSON olarak parse et
            try:
                result = response.json()
                logger.info(f"Astria API durum yanıtı: {json.dumps(result)[:100]}...")
                
                # Görsel URL'sini al
                image_url = None
                image_urls = []
                status = "processing"
                is_ready = False
                
                # Görsel URL'lerini farklı formatlarda kontrol et
                if 'images' in result and isinstance(result['images'], list) and len(result['images']) > 0:
                    for image in result['images']:
                        if isinstance(image, dict) and 'url' in image:
                            image_urls.append(image.get('url'))
                        elif isinstance(image, str):
                            image_urls.append(image)
                
                # Diğer olası formatları kontrol et
                if not image_urls and 'image_url' in result:
                    image_urls.append(result.get('image_url'))
                if not image_urls and 'output' in result and isinstance(result['output'], dict) and 'image_url' in result['output']:
                    image_urls.append(result['output']['image_url'])
                
                # İlk görsel URL'sini ana URL olarak ayarla (geriye dönük uyumluluk için)
                if image_urls:
                    image_url = image_urls[0]
                    # Store each generated image in our library
                    for url in image_urls:
                        store_generated_content(
                            url=url,
                            content_type="video-image",
                            type="image",
                            prompt=prompt,
                            model="astria-flux1-dev",
                        )
                
                # Durum bilgisini kontrol et
                if 'status' in result:
                    status = result['status']
                    # Durum "completed" ise görsel hazır demektir
                    if status.lower() in ["completed", "success", "done"]:
                        is_ready = True
                
                # Görsel URL'si varsa hazır kabul et
                if image_urls:
                    is_ready = True
                
                # Görsel URL'lerini loglama
                if image_urls:
                    logger.info(f"Toplam {len(image_urls)} görsel URL bulundu")
                    logger.info(f"İlk görsel URL: {image_urls[0]}")
                else:
                    logger.warning(f"Görsel URL bulunamadı. Yanıt: {json.dumps(result)[:200]}...")
                
                # Her durumda JSON yanıtı döndür
                return jsonify({
                    "is_ready": is_ready,
                    "status": status,
                    "image_url": image_url,  # Geriye dönük uyumluluk için
                    "image_urls": image_urls,  # Tüm görsel URL'leri
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "brand": brand,
                    "aspect_ratio": aspect_ratio  # Aspect ratio bilgisini ekle
         #            "content_type": content_type  # Add content_type to response
                })
            except json.JSONDecodeError:
                logger.error(f"Astria API yanıtı JSON formatında değil: {response.text[:100]}...")
                return jsonify({"error": "API yanıtı geçersiz format"}), 500
        else:
            logger.error(f"Astria API durum kontrolü hatası: {response.status_code} - {response.text}")
            return jsonify({"error": f"Durum kontrolü sırasında bir hata oluştu: {response.status_code}"}), response.status_code
    except Exception as e:
        logger.error(f"Durum kontrolü hatası: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_image_2', methods=['POST'])
def generate_image_2():
    prompt = request.form.get('prompt')
    brand_input = request.form.get('brand_input')
    aspect_ratio = request.form.get('aspect_ratio', '1:1')  # Varsayılan olarak 1:1
    redirect_to_page = request.form.get('redirect', 'false').lower() == 'true'  # Yönlendirme seçeneği
    content_type = request.form.get('content_type', 'video-image')  # Changed default to video-image
    
    if not prompt:
        return jsonify({"error": "Geçersiz prompt seçimi"}), 400
    
    try:
        logger.info(f"Astria AI API'sine görsel oluşturma isteği gönderiliyor")
        logger.info(f"Kullanılan prompt: {prompt[:50]}...")  # İlk 50 karakteri logla
        logger.info(f"Kullanılan aspect ratio: {aspect_ratio}")
   #      logger.info(f"Content type: {content_type}")
        
        # API URL'sini kontrol et - Flux API'sini kullanacağız
        api_key = os.getenv("ASTRIA_API_KEY")
        
        # Flux model ID - Astria'nın genel Flux modelini kullanıyoruz
        flux_model_id = "1504944"  # Flux1.dev from the gallery
        
        # API URL'sini oluştur
        api_url = f"https://api.astria.ai/tunes/{flux_model_id}/prompts"
        
        if not api_key:
            logger.error(f"Astria API bilgileri eksik. Key: {api_key[:5] if api_key else None}...")
            return jsonify({"error": "API yapılandırması eksik"}), 500
            
        logger.info(f"Astria API URL: {api_url}")
        
        # Benzersiz bir istek ID'si oluştur
        request_id = str(uuid.uuid4())
        logger.info(f"Oluşturulan istek ID: {request_id}")
        
        # Astria AI dokümantasyonuna göre boyutları ayarla
        # Boyutlar 8'in katları olmalıdır
        if aspect_ratio == "1:1":
            width, height = 1024, 1024  # Kare format
        elif aspect_ratio == "4:5":
            width, height = 1024, 1280  # Instagram post formatı
        elif aspect_ratio == "16:9":
            width, height = 1280, 720  # Yatay video/web formatı
        elif aspect_ratio == "9:16":
            width, height = 720, 1280  # Dikey story formatı
        else:
            # Varsayılan olarak 1:1 kullan
            width, height = 1024, 1024
            logger.warning(f"Bilinmeyen aspect ratio: {aspect_ratio}, varsayılan 1:1 kullanılıyor")
        
        logger.info(f"Kullanılan görsel boyutu: {width}x{height}")
        
        # Prompt'a aspect ratio bilgisini ekle ve optimize et
        # Astria AI dokümantasyonuna göre prompt'u düzenle
        aspect_ratio_prompt = ""
        if aspect_ratio == "1:1":
            aspect_ratio_prompt = "square format, 1:1 aspect ratio"
        elif aspect_ratio == "4:5":
            aspect_ratio_prompt = "portrait format, 4:5 aspect ratio, vertical composition"
        elif aspect_ratio == "16:9":
            aspect_ratio_prompt = "landscape format, 16:9 aspect ratio, horizontal composition"
        elif aspect_ratio == "9:16":
            aspect_ratio_prompt = "vertical format, 9:16 aspect ratio, portrait composition"
        
        enhanced_prompt = f"{prompt}, {aspect_ratio_prompt}, high quality, detailed"
        logger.info(f"Geliştirilmiş prompt: {enhanced_prompt[:100]}...")
        
        # Astria AI API isteği için form data hazırla
        # Dokümantasyona göre parametreleri ayarla
        data = {
            'prompt[text]': enhanced_prompt,
            'prompt[w]': str(width),
            'prompt[h]': str(height),
            'prompt[num_inference_steps]': "50",  # Daha yüksek kalite için 50 adım
            'prompt[guidance_scale]': "7.5",      # Prompt'a uyum için 7.5 değeri
            'prompt[seed]': "-1",                 # Rastgele seed
            'prompt[lora_scale]': "0.8"           # LoRA ağırlığı
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        # Payload'ı logla (hassas bilgileri gizleyerek)
        logger.info(f"Astria API data: {json.dumps(data)}")
        
        # İstek zamanını ölç
        request_start_time = time.time()
        logger.info("Astria AI isteği başlıyor...")
        
        # Astria AI API'sine istek gönder
        response = requests.post(
            api_url,
            headers=headers,
            data=data
        )
        
        # İstek süresini hesapla
        request_duration = time.time() - request_start_time
        logger.info(f"Astria AI isteği tamamlandı. Süre: {request_duration:.2f} saniye")
        logger.info(f"Astria API yanıt kodu: {response.status_code}")
        
        # Yanıtı kontrol et
        if response.status_code == 200 or response.status_code == 201:
            try:
                result = response.json()
                logger.info(f"Astria AI yanıtı başarılı: {json.dumps(result)[:100]}...")
            except json.JSONDecodeError:
                # Yanıt JSON değilse, metin olarak al
                result = response.text
                logger.warning(f"Astria API yanıtı JSON formatında değil: {result[:100]}...")
                return jsonify({
                    "error": "API yanıtı geçersiz format",
                    "details": result[:200] + "..." if len(result) > 200 else result
                }), 500
            
            # Görsel URL'sini al - Astria API'sinin yanıt formatına göre
            image_url = None
            image_urls = []
            
            # Yanıt formatını kontrol et
            if isinstance(result, dict):
                # Prompt ID'yi kontrol et
                prompt_id = result.get('id')
                
                # Görsel URL'lerini farklı formatlarda kontrol et
                if 'images' in result and isinstance(result['images'], list) and len(result['images']) > 0:
                    for image in result['images']:
                        if isinstance(image, dict) and 'url' in image:
                            image_urls.append(image.get('url'))
                        elif isinstance(image, str):
                            image_urls.append(image)
                
                # Diğer olası formatları kontrol et
                if not image_urls and 'image_url' in result:
                    image_urls.append(result.get('image_url'))
                if not image_urls and 'output' in result and isinstance(result['output'], dict) and 'image_url' in result['output']:
                    image_urls.append(result['output']['image_url'])
                
                # İlk görsel URL'sini ana URL olarak ayarla (geriye dönük uyumluluk için)
                if image_urls:
                    image_url = image_urls[0]
                    # Store each generated image in our library
                    for url in image_urls:
                        store_generated_content(
                           url=url,
                            content_type= content_type,  # Use the content_type from request
                            type="image",
                            prompt=prompt,
                            model="astria-flux1-dev",
                        )
                
                # Görsel URL'lerini loglama
                if image_urls:
                    logger.info(f"Toplam {len(image_urls)} görsel URL bulundu")
                    logger.info(f"İlk görsel URL: {image_urls[0]}")
                else:
                    logger.warning(f"Görsel URL bulunamadı. Yanıt: {json.dumps(result)[:200]}...")
                
                if not image_urls:
                    logger.error("Astria AI yanıtında görsel URL'si bulunamadı")
                    logger.error(f"Tam yanıt: {json.dumps(result)}")
                    
                    # Prompt ID varsa, asenkron işleme için döndür
                    if prompt_id:
                        logger.info(f"Prompt ID bulundu: {prompt_id}. Görsel hazır olduğunda kontrol edilebilir.")
                        return jsonify({
                            "success": True,
                            "prompt_id": prompt_id,
                            "prompt": prompt,
                            "aspect_ratio": aspect_ratio,
                            "request_id": request_id,
                            "message": "Görsel asenkron olarak oluşturuluyor. Lütfen birkaç dakika sonra tekrar kontrol edin."
                        })
                    
                    return jsonify({"error": "Görsel oluşturulamadı", "details": result}), 500
                
                # Eğer yönlendirme isteniyorsa, image.html sayfasına yönlendir
                if redirect_to_page:
                    return redirect(url_for('image', image_url=image_urls, prompt=prompt, brand=brand_input))
                
                # Aksi takdirde JSON yanıtı döndür
                return jsonify({
                    "success": True,
                    "image_url": image_url,  # Geriye dönük uyumluluk için
                    "image_urls": image_urls,  # Tüm görsel URL'leri
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "request_id": request_id,
                    "prompt_id": prompt_id  # Prompt ID'yi de döndür
                })
            else:
                logger.error(f"Beklenmeyen yanıt formatı: {type(result)}")
                return jsonify({"error": "Beklenmeyen yanıt formatı", "details": str(result)[:200]}), 500
        else:
            logger.error(f"Astria AI API hatası: {response.status_code} - {response.text}")
            return jsonify({
                "error": f"Görsel oluşturulurken bir hata oluştu: {response.status_code}",
                "details": response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Görsel oluşturma hatası: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Görsel oluşturulurken bir hata oluştu: {str(e)}"}), 500

@app.route("/image-to-video", methods=["POST"])
def image_to_video():
    """Görüntüyü videoya dönüştürmek için API endpoint'i."""
    try:
        data = request.get_json()
        if not data:
            logger.error("Request body is empty or not JSON")
            return jsonify({"error": "Invalid request format. JSON body required."}), 400
            
        image_url = data.get("image_url")
        content_type = data.get("content_type", "video-image")
        
        if not image_url:
            logger.error("Missing required parameter 'image_url' in image_to_video")
            return jsonify({"error": "Missing required parameter: 'image_url'"}), 400
            
        if not content_type:
            logger.error("Missing required parameter 'content_type' in image_to_video")
            return jsonify({"error": "Missing required parameter: 'content_type'"}), 400
            
        if content_type not in ['product-visual', 'video-image']:
            logger.error(f"Invalid content type: {content_type}")
            return jsonify({"error": "Invalid content type. Must be 'product-visual' or 'video-image'"}), 400

        duration = data.get("duration")
        if not duration:
            logger.error("Missing required parameter 'duration' in image_to_video")
            return jsonify({"error": "Missing required parameter: 'duration'"}), 400

        # Video oluşturma işlemi
        try:
            prompt = "a video with motion, maintaining the exact same style, colors, and composition as the original image."
            request_id = str(uuid.uuid4())
            
            # Replicate API'ye gönderilecek input parametreleri
            input_data = {
                "start_image_url": image_url,
                "prompt": prompt,
                "num_frames": 90,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "motion_bucket_id": 127,
                "noise_aug_strength": 0.02,
                "cond_aug": 0.02,
                "seed": random.randint(1, 1000000),
                "duration": duration
            }

            output = replicate.run(
                "luma/ray-flash-2-720p",
                input=input_data
            )
            
            # FileOutput nesnesini string'e dönüştür
            try:
                video_url = str(output)
                if not video_url.startswith('http'):
                    logger.error(f"Invalid video URL format: {video_url}")
                    return jsonify({"error": "Invalid video URL format in API response"}), 500
            except Exception as e:
                logger.error(f"Error converting FileOutput to URL: {str(e)}")
                return jsonify({"error": "Failed to process video URL from API response"}), 500
            
            # First, store the video in Supabase storage
            logger.info(f"Storing video in Supabase storage: {video_url}")
            supabase_url = store_in_supabase(
                url=video_url,
                content_type=content_type,
                type="video"
            )
            
            # Then store metadata in the database (use Supabase URL if successful, otherwise original URL)
            store_generated_content(
                url=supabase_url if supabase_url else video_url,
                content_type=content_type,
                type="video",
                prompt=prompt,
                model="luma/ray-flash-2-720p",
                duration=duration
            )
            
            # Return both the original and Supabase URLs
            response_data = {
                "success": True,
                "video_url": supabase_url if supabase_url else video_url,
                "original_url": video_url,
                "request_id": request_id,
                "prompt": prompt,
                "stored_in_supabase": supabase_url is not None and supabase_url != video_url
            }
            
            return jsonify(response_data)
            
        except Exception as api_error:
            logger.error(f"Replicate API error: {str(api_error)}")
            return jsonify({"error": f"Video generation failed: {str(api_error)}"}), 500
            
    except Exception as e:
        logger.error(f"General error in image-to-video endpoint: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/library')
def library():
    """Display the content library page with data from Supabase."""
    try:
        logger.info("Library route accessed")
        
        # Check if Supabase is initialized
        if not supabase:
            logger.error("Supabase client not initialized in library route. Cannot fetch data.")
            return render_template('library.html', media_items=[], error="Database connection not available")
        
        # Fetch all data from Supabase (using a high limit to get everything)
        logger.info("Calling fetch_generations_from_db to get all records...")
        media_items = fetch_generations_from_db(limit=10000, offset=0)
        
        # Log the data being sent to template
        logger.info(f"Fetch completed. Retrieved {len(media_items)} items for library template")
        
        if not media_items:
            logger.warning("No media items found in the database")
            
        for i, item in enumerate(media_items[:5]):  # Log only the first 5 items to avoid overly large logs
            logger.info(f"Item {i+1}: id={item.get('id')}, type={item.get('type')}, content_type={item.get('content_type')}, url={item.get('url')[:50]}")
        
        if len(media_items) > 5:
            logger.info(f"... and {len(media_items) - 5} more items")
        
        return render_template('library.html', media_items=media_items)
    except Exception as e:
        logger.error(f"Error in library route: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return render_template('library.html', media_items=[], error=str(e))

@app.errorhandler(404)
def page_not_found(e):
    logger.error(f"404 error: {str(e)}")
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"500 error: {str(e)}")
    return render_template('error.html', error="Internal server error"), 500

@app.route('/delete-generation/<generation_id>', methods=['DELETE'])
def delete_generation(generation_id):
    """Delete a generation from the database."""
    try:
        if not supabase:
            return jsonify({"error": "Database connection not available"}), 500
            
        # Validate UUID format
        try:
            # Convert string to UUID to validate format
            uuid_obj = uuid.UUID(generation_id)
        except ValueError:
            logger.error(f"Invalid UUID format: {generation_id}")
            return jsonify({"error": "Invalid generation ID format"}), 400
            
        # Delete the record
        response = supabase.table('generations') \
            .delete() \
            .eq('id', str(uuid_obj)) \
            .execute()
            
        if response and response.data:
            logger.info(f"Successfully deleted generation {generation_id}")
            return jsonify({"success": True})
        else:
            logger.error(f"No generation found with ID {generation_id}")
            return jsonify({"error": "Generation not found"}), 404
            
    except Exception as e:
        logger.error(f"Error deleting generation {generation_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Add these global variables near the top of the file, after the imports and before the routes
# Credit cache to reduce Supabase queries
last_credit_check = None
credit_cache = None
CREDIT_CACHE_TTL = 30  # seconds

@app.route('/api/get-credit', methods=['GET'])
def get_credit():
    """API endpoint to fetch current credit information from Supabase with caching."""
    global last_credit_check, credit_cache
    
    try:
        logger.info("Credit information requested")
        
        # Check if we have a valid cached response
        current_time = time.time()
        if last_credit_check and credit_cache and current_time - last_credit_check < CREDIT_CACHE_TTL:
            logger.info("Returning cached credit information")
            return jsonify(credit_cache)
        
        if not supabase:
            logger.error("Supabase client not initialized. Cannot fetch credit information.")
            return jsonify({"error": "Database connection not available"}), 500
        
        # Query Supabase for the latest credit information
        try:
            response = supabase.table('generations') \
                .select('id, credit, created_at') \
                .order('created_at', desc=True) \
                .limit(1) \
                .execute()
            
            if response and response.data and len(response.data) > 0:
                credit_info = response.data[0]
                logger.info(f"Credit info retrieved: {credit_info}")
                
                # Prepare the response
                result = {
                    "success": True,
                    "credit": credit_info.get('credit', 0),
                    "id": credit_info.get('id'),
                    "created_at": credit_info.get('created_at')
                }
                
                # Update cache
                last_credit_check = current_time
                credit_cache = result
                
                return jsonify(result)
            else:
                logger.warning("No credit records found in database")
                result = {
                    "success": True, 
                    "credit": 0,
                    "message": "No credit records found"
                }
                
                # Update cache even for empty results
                last_credit_check = current_time
                credit_cache = result
                
                return jsonify(result)
                
        except Exception as e:
            logger.error(f"Error querying credit from Supabase: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Failed to query credit: {str(e)}"}), 500
        
    except Exception as e:
        logger.error(f"Error in get_credit endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/listen-credit-changes', methods=['GET'])
def listen_credit_changes():
    """
    Server-sent events endpoint for credit changes.
    This establishes a long-lived connection to notify the client of credit changes.
    """
    def generate():
        global last_credit_check, credit_cache
        last_credit_id = None
        client_connected = True
        
        # Log when a client connects
        logger.info("SSE client connected for credit updates")
        
        while client_connected:
            try:
                if supabase:
                    response = supabase.table('generations') \
                        .select('id, credit, created_at') \
                        .order('created_at', desc=True) \
                        .limit(1) \
                        .execute()
                    
                    if response and response.data and len(response.data) > 0:
                        credit_info = response.data[0]
                        current_id = credit_info.get('id')
                        
                        # Send data if it's the first time or if the credit has changed
                        if last_credit_id is None or last_credit_id != current_id:
                            # Update the cache with the new value
                            current_time = time.time()
                            last_credit_check = current_time
                            credit_cache = {
                                "success": True,
                                "credit": credit_info.get('credit', 0),
                                "id": credit_info.get('id'),
                                "created_at": credit_info.get('created_at')
                            }
                            
                            yield f"data: {json.dumps(credit_info)}\n\n"
                            last_credit_id = current_id
                            logger.info(f"Sent updated credit info: {credit_info}")
            except Exception as e:
                logger.error(f"Error sending credit data: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
            # Send a heartbeat to check if client is still connected
            try:
                yield f": heartbeat\n\n"  # Comment line as heartbeat
            except Exception:
                # If we can't send data, client has disconnected
                client_connected = False
                logger.info("SSE client disconnected from credit updates")
                break
            
            # Check for updates less frequently (15 seconds instead of 5)
            time.sleep(15)
    
    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    return response

@app.route('/api/recent-content', methods=['GET'])
def get_recent_content():
    try:
        limit = request.args.get('limit', 6, type=int)
        page = request.args.get('page', 1, type=int)
        offset = (page - 1) * limit
        
        # Use Supabase to fetch recent content
        if not supabase:
            logger.error("Supabase client not initialized. Cannot fetch recent content.")
            return jsonify({
                'success': False,
                'error': "Database connection not available"
            }), 500
            
        try:
            # Build the base query
            query = supabase.table('generations') \
                .select('*') \
                .order('created_at', desc=True)
            
            # Handle pagination
            try:
                # Check if offset method is available
                if hasattr(query, 'offset'):
                    # Use standard pagination with offset
                    response = query.limit(limit).offset(offset).execute()
                else:
                    # Manual pagination approach
                    fetch_limit = offset + limit
                    response = query.limit(fetch_limit).execute()
                    
                    # Apply offset and limit manually by slicing the results
                    all_data = response.data if response.data else []
                    response.data = all_data[offset:offset+limit] if len(all_data) > offset else []
            except Exception as e:
                logger.warning(f"Pagination error: {str(e)}")
                # Fallback to simpler approach
                response = query.limit(limit).execute()
                all_data = response.data if response.data else []
                response.data = all_data[offset:offset+limit] if len(all_data) > offset else []
            
            # Process the results
            items = []
            for item in response.data:
                view_url = None
                if item.get('type') == 'image':
                    view_url = f'/image?id={item.get("id")}'
                elif item.get('type') == 'video':
                    view_url = f'/video?id={item.get("id")}'
                elif item.get('type') == 'image2':
                    view_url = f'/image2?id={item.get("id")}'
                    
                items.append({
                    'id': item.get('id'),
                    'url': item.get('url'),
                    'type': item.get('type', ''),
                    'title': item.get('content_type', ''),
                    'created_at': item.get('created_at'),
                    'view_url': view_url or ''
                })
            
            # Get total count for pagination or estimate
            try:
                count_response = supabase.table('generations').select('id', count='exact').execute()
                total_count = count_response.count if hasattr(count_response, 'count') else len(items)
            except Exception:
                # Use length of current items as fallback
                total_count = len(items)
            
            # Calculate if more items are available
            has_more = total_count > (offset + limit) or (len(items) == limit and total_count <= 0)
            
            return jsonify({
                'success': True,
                'items': items,
                'page': page,
                'limit': limit,
                'total': total_count,
                'has_more': has_more
            })
            
        except Exception as e:
            logger.error(f"Error querying Supabase: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f"Database error: {str(e)}"
            }), 500
            
    except Exception as e:
        logger.error(f"Error in get_recent_content: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/add-audio-to-video', methods=['POST'])
def add_audio_to_video():
    """Uploads a video to fal-ai/mmaudio-v2 to add audio based on a prompt."""
    if not FAL_CLIENT_AVAILABLE:
        logger.error("fal.ai client kütüphanesi yüklü değil")
        return jsonify({"error": "fal.ai client kütüphanesi yüklü değil"}), 500
    try:
        data = request.get_json()
        if not data:
            logger.error("Request body is empty or not JSON")
            return jsonify({"error": "Invalid request format. JSON body required."}), 400
        video_url = data.get("video_url")
        prompt = data.get("prompt")
        content_type = data.get("content_type") 
        if not video_url or not prompt:
            logger.error("Missing required parameters: video_url and prompt")
            return jsonify({"error": "Missing required parameters: 'video_url' and 'prompt'"}), 400
        logger.info(f"Uploading video to fal-ai/mmaudio-v2: {video_url} with prompt: {prompt}")
        fal_client.api_key = os.getenv('FAL_KEY')
        def on_queue_update(update):
            if hasattr(update, 'logs') and update.logs:
                for log in update.logs:
                    logger.info(f"[fal-ai/mmaudio-v2] {log.get('message', '')}")
        try:
            result = fal_client.subscribe(
                "fal-ai/mmaudio-v2",
                arguments={
                    "video_url": video_url,
                    "prompt": prompt
                },
                with_logs=True,
                on_queue_update=on_queue_update,
            )
            logger.info(f"fal-ai/mmaudio-v2 result: {result}")
            # Extract and save MMAudio output URL
            audio_video_url = None
            if isinstance(result, dict) and 'video' in result and isinstance(result['video'], dict) and 'url' in result['video']:
                audio_video_url = result['video']['url']
                logger.info(f"Successfully extracted video URL from result: {audio_video_url}")
            else:
                logger.error(f"Could not extract video URL from result. Result structure: {result}")
                return jsonify({"error": "Could not extract video URL from API response"}), 500
            
            if audio_video_url:
                try:
                    store_generated_content(
                        url=audio_video_url,
                        content_type= content_type,
                        type="audio",
                        prompt=prompt,
                        model="fal-ai/mmaudio-v2"
                    )
                    logger.info(f"Audio-video URL veritabanına kaydedildi: {audio_video_url}")
                except Exception as e:
                    logger.error(f"Audio-video URL kaydedilemedi: {str(e)}")
            else:
                logger.error("Audio-video URL bulunamadı; veritabanı kaydı atlanıyor.")
            return jsonify({"success": True, "audio_video_url": audio_video_url, "video_url": audio_video_url, "result": result})
        except Exception as fal_error:
            logger.error(f"fal-ai/mmaudio-v2 API error: {str(fal_error)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"fal-ai/mmaudio-v2 API error: {str(fal_error)}"}), 500
    except Exception as e:
        logger.error(f"General error in add_audio_to_video endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    
def store_in_supabase(url, content_type, type):
    """REST API yöntemiyle Supabase Storage'a yükler"""
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
        
        project_ref = os.getenv("SUPABASE_PROJECT_REF")
        anon_key = os.getenv("SUPABASE_KEY")
        
        if not project_ref or not anon_key:
            logger.error("❌ Missing Supabase credentials in environment variables")
            return url
            
        logger.info(f"🔑 Using Supabase credentials from environment variables")
        
        bucket_id = "generations"
        
        # Upload URL'sini doğru formatta tanımla
        upload_url = f"https://{project_ref}.supabase.co/storage/v1/object/{bucket_id}/{filename}"
        content_type_header = "video/mp4" if type == "video" else "image/jpeg"
        logger.info(f"🌐 Upload URL: {upload_url}")
        logger.info(f"📊 Content-Type: {content_type_header}")
        
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
        
        # Generate the URL for the uploaded content
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

if __name__ == '__main__':
    logger.info("Uygulama başlatılıyor...")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)