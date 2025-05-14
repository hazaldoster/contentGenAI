#!/bin/bash
set -e

echo "🚀 Starting deployment process..."

# Ensure we're using the correct version
echo "📋 Checking requirements.txt for fal-client..."
if ! grep -q "fal-client==0.5.9" requirements.txt; then
  echo "❌ fal-client==0.5.9 not found in requirements.txt"
  echo "Adding fal-client to requirements.txt..."
  sed -i 's/# fal-client==0.1.0 may be causing issues/fal-client==0.5.9/' requirements.txt
fi

# Run test script locally first
echo "🧪 Running fal.ai client test locally..."
python test_fal_client.py

# Check if test was successful
if [ $? -ne 0 ]; then
  echo "❌ Local test failed. Please fix the issues before deploying."
  exit 1
fi

# Commit changes
echo "💾 Committing changes..."
git add requirements.txt Dockerfile test_fal_client.py
git commit -m "Fix fal.ai client integration with version 0.5.9"

# Deploy to Railway
echo "🚂 Deploying to Railway..."
# If railway CLI is installed
if command -v railway &> /dev/null; then
  railway up
else
  echo "ℹ️ Railway CLI not installed. Please deploy manually with 'railway up' or through the Railway dashboard."
  echo "You can install Railway CLI with: npm i -g @railway/cli"
fi

echo "✅ Deployment process completed!"
echo "🌐 Please test the application once deployment is complete."
echo "Run 'python test_fal_client.py' on the server to verify fal.ai integration." 