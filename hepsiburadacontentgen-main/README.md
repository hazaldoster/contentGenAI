# HepsiBurada Content Generator

AI-powered content creation tool for e-commerce platforms. Generate visual and video content from text input.

## Features

- Generate images and videos from text
- Multiple style options with GPT-4o
- High-quality images with Astria AI
- Dynamic videos with Veo2
- Various aspect ratios for different platforms
- Brand-compatible content options

## Technologies

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript, Tailwind CSS
- **AI**: OpenAI GPT-4o, Astria AI, Fal.ai

## Quick Start

1. Clone and install dependencies:
   ```bash
   git clone https://github.com/yourusername/hepsiburadacontentgen.git
   cd hepsiburadacontentgen
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key
   FAL_API_KEY=your_fal_api_key
   ASSISTANT_ID=your_assistant_id_if_any
   ASTRIA_API_KEY=your_astria_api_key
   ```

3. Run the app:
   ```bash
   python app.py
   ```

## Deploying on Vercel

1. Install and login to Vercel CLI:
   ```bash
   npm install -g vercel
   vercel login
   ```

2. Deploy:
   ```bash
   vercel
   ```

3. When prompted for directory, enter `.` (dot)

4. Set environment variables in Vercel dashboard
