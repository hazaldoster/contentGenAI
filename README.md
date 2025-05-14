# Agentic Content Generator
an agentic AI project that creates txt2img, txt2vid, img2vid content powered by LLM prompting and supported audio. AI-powered content creation tool for e-commerce platforms.

## Features

- **Text-to-Image Generation**: Create professional product images from text descriptions
- **Image-to-Video Conversion**: Transform static images into dynamic videos
- **Multiple Style Options**: Select from various visual styles powered by GPT-4o
- **Brand Consistency**: Generate content that matches your brand guidelines
- **Multiple Aspect Ratios**: Support for different platforms (1:1, 16:9, 9:16, 4:5)
- **Content Library**: Browse and manage previously generated content
- **Audio Addition**: Add audio tracks to generated videos

## AI Technologies

- **OpenAI GPT-4o**: Intelligent prompt generation and style detection
- **Astria AI**: High-quality image generation
- **Fal.ai**: Advanced image processing and transformation
- **Replicate**: Video generation capabilities

## Tech Stack

- **Backend**: Flask (Python 3.9)
- **Frontend**: HTML, CSS, JavaScript, Tailwind CSS
- **Database**: Supabase
- **Deployment**: Docker, Railway/Vercel compatible

## Setup & Installation

### Prerequisites

- Python 3.9+
- Node.js and npm
- API keys for OpenAI, Fal.ai, Astria AI, and Replicate

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hepsiburadacontentgen.git
   cd hepsiburadacontentgen
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Node.js dependencies:
   ```bash
   npm install
   ```

4. Create a `.env` file with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   FAL_API_KEY=your_fal_api_key
   ASTRIA_API_KEY=your_astria_api_key
   REPLICATE_API_KEY=your_replicate_api_key
   SECRET_KEY=your_flask_secret_key
   ```

5. Run the application:
   ```bash
   python app.py
   ```

6. Access the application at http://localhost:5000

### Docker Deployment

Build and run with Docker:

```bash
docker build -t hepsiburada-content-gen .
docker run -p 5000:5000 --env-file .env hepsiburada-content-gen
```

### Railway Deployment

This project includes a `railway.json` file for easy deployment on Railway.

1. Install Railway CLI and login:
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. Deploy the application:
   ```bash
   railway up
   ```

3. Set environment variables in the Railway dashboard

## API Endpoints

- `/generate-prompt`: Generate optimized prompts for AI models
- `/generate_image`: Create images with various AI models
- `/generate_video`: Create videos from prompts or images
- `/image-to-video`: Convert static images to videos
- `/add-audio-to-video`: Add audio tracks to videos
- `/api/recent-content`: Get recently generated content
