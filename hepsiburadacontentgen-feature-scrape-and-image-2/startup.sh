#!/bin/bash
set -e

echo "Installing any missing dependencies..."
pip install --no-cache-dir -r requirements.txt || echo "Some dependencies may be missing but we'll try to run anyway"

echo "Setting up Supabase environment variables..."
if [ -f ".env.supabase" ]; then
  export $(cat .env.supabase | xargs)
  echo "Loaded Supabase environment variables from .env.supabase"
else
  echo "Warning: .env.supabase file not found"
fi

# Check Supabase environment variables
if [ -z "$SUPABASE_URL" ]; then
  echo "Warning: SUPABASE_URL is not set"
  export SUPABASE_URL="https://vsczjwvmkqustdbxyvzo.supabase.co"
  echo "Set default SUPABASE_URL"
fi

if [ -z "$SUPABASE_KEY" ]; then
  echo "Warning: SUPABASE_KEY is not set"
  export SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZzY3pqd3Zta3F1c3RkYnh5dnpvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDE4NzU5NjQsImV4cCI6MjA1NzQ1MTk2NH0.7tlRgk0sPXHZnmbnvPyOkEHT-ptJMK8BGvINY-5YPds"
  echo "Set default SUPABASE_KEY"
fi

# Print environment info
echo "SUPABASE_URL is ${SUPABASE_URL:0:15}..."
echo "SUPABASE_KEY is ${SUPABASE_KEY:0:15}..."

echo "Starting application..."
python app.py 