FROM python:3.9-slim

WORKDIR /app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    build-essential \
    libffi-dev \
    python3-dev \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies with better error handling
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt || \
    (echo "Failed to install all requirements at once, trying one by one" && \
     pip install flask==2.3.3 && \
     pip install requests==2.31.0 && \
     pip install python-dotenv==1.0.0 && \
     pip install urllib3==1.26.15 && \
     pip install Pillow==9.5.0 && \
     pip install gunicorn==21.2.0 && \
     pip install openai==1.3.0 && \
     pip install beautifulsoup4==4.12.2 && \
     pip install supabase==1.0.3 && \
     pip install fal-client==0.5.9 && \
     pip install replicate==0.22.0)

# Copy package.json and install Node dependencies
COPY package.json .
RUN npm install

# Copy the rest of the application
COPY . .

# Make the startup script executable
COPY startup.sh .
RUN chmod +x startup.sh

# Expose the port the app runs on
EXPOSE 5000

# Start the application
CMD ["./startup.sh"] 