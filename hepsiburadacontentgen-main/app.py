from flask import Flask, render_template, request, jsonify, redirect, url_for
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

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log Python version and environment
logger.info(f"Python version: {sys.version}")
logger.info(f"Environment: {os.environ.get('VERCEL_ENV', 'local')}")

# Fal.ai client kütüphanesini içe aktar - hata yönetimi ile
try:
    import fal_client
    FAL_CLIENT_AVAILABLE = True
    logger.info("fal_client kütüphanesi başarıyla yüklendi.")
except ImportError as e:
    FAL_CLIENT_AVAILABLE = False
    logger.warning(f"fal_client kütüphanesi yüklenemedi: {str(e)}. Video oluşturma özellikleri devre dışı olacak.")

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

# API anahtarları
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FAL_API_KEY = os.getenv("FAL_API_KEY")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
ASTRIA_API_URL = os.getenv("ASTRIA_API_URL")
ASTRIA_API_KEY = os.getenv("ASTRIA_API_KEY")

# Log API key availability (not the actual keys)
logger.info(f"OPENAI_API_KEY mevcut: {bool(OPENAI_API_KEY)}")
logger.info(f"FAL_API_KEY mevcut: {bool(FAL_API_KEY)}")
logger.info(f"ASTRIA_API_KEY mevcut: {bool(ASTRIA_API_KEY)}")

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
for template_name in ['welcome.html', 'index.html', 'image.html', 'video.html']:
    template_path = os.path.join(template_dir, template_name)
    if not os.path.exists(template_path):
        logger.warning(f"Template {template_name} not found. Creating a basic version.")
        try:
            with open(template_path, 'w') as f:
                f.write(f"<!DOCTYPE html><html><head><title>{template_name}</title></head><body><h1>{template_name}</h1><p>This is a placeholder template.</p></body></html>")
        except Exception as e:
            logger.error(f"Failed to create template {template_name}: {str(e)}")

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

def generate_prompt(text: str, feature_type: str, aspect_ratio: str = "1:1") -> dict:
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

@app.route('/')
def welcome():
    """Karşılama sayfasını göster"""
    logger.info("Karşılama sayfası görüntüleniyor")
    return render_template('welcome.html')

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
    prompt_id = request.args.get('prompt_id')
    
    # Eğer prompt_id varsa ve görsel URL'leri yoksa, check_image_status fonksiyonunu çağır
    if prompt_id and not image_urls:
        try:
            # API bilgilerini al
            api_key = os.getenv("ASTRIA_API_KEY")
            
            if not api_key:
                logger.error("API yapılandırması eksik")
                return render_template('image.html')
            
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
        return render_template('image.html')
    
    logger.info(f"Görsel sonuç sayfası görüntüleniyor. Görsel URL sayısı: {len(image_urls)}")
    return render_template('image.html', image_urls=image_urls, prompt=prompt, brand=brand, prompt_id=prompt_id)

@app.route("/generate-prompt", methods=["POST"])
def generate_prompt_api():
    """API endpoint for generating prompts."""
    data = request.json
    text = data.get("text")
    feature_type = data.get("feature_type")
    aspect_ratio = data.get("aspect_ratio", "1:1")  # Varsayılan olarak 1:1
    
    if not text or not feature_type:
        return jsonify({"error": "Missing required parameters: 'text' and 'feature_type'"}), 400
    
    try:
        result = generate_prompt(text, feature_type, aspect_ratio)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/generate_video', methods=['POST'])
def generate_video():
    prompt = request.form.get('prompt')
    brand_input = request.form.get('brand_input')
    aspect_ratio = request.form.get('aspect_ratio', '9:16')  # Varsayılan olarak 9:16
    duration = request.form.get('duration', '5s')  # Yeni: Video süresi parametresi
    
    if not prompt:
        return jsonify({"error": "Geçersiz prompt seçimi"}), 400
    
    # Fal.ai client'ın kullanılabilir olup olmadığını kontrol et
    if not FAL_CLIENT_AVAILABLE:
        logger.error("fal_client kütüphanesi yüklü değil. Video oluşturulamıyor.")
        return jsonify({"error": "Video oluşturma özelliği şu anda kullanılamıyor. Sunucu yapılandırması eksik."}), 500
    
    try:
        logger.info(f"Fal.ai API'sine video oluşturma isteği gönderiliyor")
        logger.info(f"Kullanılan prompt: {prompt[:50]}...")
        logger.info(f"Kullanılan aspect ratio: {aspect_ratio}")
        logger.info(f"Kullanılan video süresi: {duration}")
        
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
                        logger.info(f"Fal.ai log: {log.get('message', '')}")
                
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
            
            # Video URL'sini test et
            logger.info("Video URL'si test ediliyor...")
            try:
                video_test = requests.head(video_url, timeout=10)
                logger.info(f"Video URL'si test sonucu: {video_test.status_code}")
                if video_test.status_code != 200:
                    logger.warning(f"Video URL'si erişilebilir değil: {video_test.status_code}")
            except Exception as video_test_error:
                logger.warning(f"Video URL'si test edilirken hata oluştu: {str(video_test_error)}")
            
            # Video sayfasına yönlendir
            logger.info("İstemciye yanıt gönderiliyor...")
            return jsonify({
                "video_url": video_url,
                "prompt": prompt,
                "brand_input": brand_input
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
                
                # API isteği için veri
                payload = {
                    "input": {
                        "prompt": prompt,
                        "aspect_ratio": aspect_ratio,  # Kullanıcının seçtiği aspect ratio
                        "duration": duration  # Kullanıcının seçtiği süre
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
                
                # Video sayfasına yönlendir
                return jsonify({
                    "video_url": video_url,
                    "prompt": prompt,
                    "brand_input": brand_input
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
    """İstek durumunu kontrol etmek için API endpoint'i"""
    # Fal.ai client'ın kullanılabilir olup olmadığını kontrol et
    if not FAL_CLIENT_AVAILABLE:
        logger.error("fal_client kütüphanesi yüklü değil. Durum kontrolü yapılamıyor.")
        return jsonify({"error": "Durum kontrolü özelliği şu anda kullanılamıyor. Sunucu yapılandırması eksik."}), 500
        
    try:
        logger.info(f"İstek durumu kontrol ediliyor (ID: {request_id})...")
        status = fal_client.status("fal-ai/veo2", request_id, with_logs=True)
        
        # Durum bilgisini JSON olarak döndür
        return jsonify({
            "status": status,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"İstek durumu kontrol edilirken hata oluştu: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_image', methods=['POST'])
def generate_image():
    prompt = request.form.get('prompt')
    brand_input = request.form.get('brand_input')
    aspect_ratio = request.form.get('aspect_ratio', '1:1')  # Varsayılan olarak 1:1
    redirect_to_page = request.form.get('redirect', 'false').lower() == 'true'  # Yönlendirme seçeneği
    
    if not prompt:
        return jsonify({"error": "Geçersiz prompt seçimi"}), 400
    
    try:
        logger.info(f"Astria AI API'sine görsel oluşturma isteği gönderiliyor")
        logger.info(f"Kullanılan prompt: {prompt[:50]}...")  # İlk 50 karakteri logla
        logger.info(f"Kullanılan aspect ratio: {aspect_ratio}")
        
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

@app.route('/test_astria_api', methods=['GET'])
def test_astria_api():
    """Astria API bağlantısını test etmek için kullanılan endpoint"""
    try:
        # API bilgilerini al
        api_key = os.getenv("ASTRIA_API_KEY")
        
        # Flux model ID - Astria'nın genel Flux modelini kullanıyoruz
        flux_model_id = "1504944"  # Flux1.dev from the gallery
        
        # API URL'sini oluştur
        api_url = f"https://api.astria.ai/tunes/{flux_model_id}/prompts"
        
        # API bilgilerini kontrol et
        if not api_key:
            return jsonify({
                "success": False,
                "error": "API bilgileri eksik",
                "api_key_exists": bool(api_key)
            })
        
        # Test için form data oluştur
        data = {
            'prompt[text]': "Test image of a blue sky with clouds",
            'prompt[width]': "896",
            'prompt[height]': "1152",
            'prompt[num_inference_steps]': "30",
            'prompt[guidance_scale]': "7.5",
            'prompt[seed]': "-1",  # Rastgele seed
            'prompt[lora_scale]': "0.8"  # LoRA ağırlığı
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        # API'ye istek gönder
        logger.info(f"Astria API test isteği gönderiliyor: {api_url}")
        response = requests.post(
            api_url,
            headers=headers,
            data=data
        )
        
        # Yanıtı kontrol et
        if response.status_code == 200 or response.status_code == 201:
            result = response.json()
            return jsonify({
                "success": True,
                "status_code": response.status_code,
                "response_preview": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            })
        else:
            return jsonify({
                "success": False,
                "status_code": response.status_code,
                "error": response.text
            })
            
    except Exception as e:
        logger.error(f"Astria API test hatası: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.route('/check_image_status/<prompt_id>', methods=['GET'])
def check_image_status(prompt_id):
    """Asenkron görsel oluşturma işleminin durumunu kontrol etmek için kullanılan endpoint"""
    try:
        # Yönlendirme seçeneğini kontrol et
        redirect_to_page = request.args.get('redirect', 'false').lower() == 'true'
        prompt = request.args.get('prompt', '')
        brand = request.args.get('brand', '')
        aspect_ratio = request.args.get('aspect_ratio', '1:1')  # Aspect ratio bilgisini al
        
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

@app.route('/debug')
def debug():
    """Debug endpoint to check environment variables and configuration"""
    debug_info = {
        "python_version": sys.version,
        "environment": os.environ.get('VERCEL_ENV', 'local'),
        "openai_api_key_exists": bool(OPENAI_API_KEY),
        "fal_api_key_exists": bool(FAL_API_KEY),
        "astria_api_key_exists": bool(ASTRIA_API_KEY),
        "fal_client_available": FAL_CLIENT_AVAILABLE,
        "template_dir_exists": os.path.exists(template_dir),
        "templates": [f for f in os.listdir(template_dir) if os.path.isfile(os.path.join(template_dir, f))] if os.path.exists(template_dir) else []
    }
    return jsonify(debug_info)

# Add error handlers
@app.errorhandler(404)
def page_not_found(e):
    logger.error(f"404 error: {str(e)}")
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"500 error: {str(e)}")
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    logger.info("Uygulama başlatılıyor...")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)