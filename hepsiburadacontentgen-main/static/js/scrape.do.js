// Scrape.do Integration
class ImageExtractor {
    constructor() {
        // No need to store API key in the frontend, it's handled by the backend
    }

    async extractImagesFromUrl(url) {
        try {
            const response = await fetch('/extract-images', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    url: url
                })
            });

            if (!response.ok) {
                throw new Error('Failed to extract images');
            }

            const data = await response.json();
            return data.product_images; // Updated to match the backend response format
        } catch (error) {
            console.error('Error extracting images:', error);
            throw error;
        }
    }
    
    // Get prompt data for generated images
    async getPromptData(imageUrl) {
        try {
            const response = await fetch('/extract-images', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    url: imageUrl
                })
            });

            if (!response.ok) {
                throw new Error('Failed to generate prompts');
            }

            const data = await response.json();
            return data.prompt_data;
        } catch (error) {
            console.error('Error generating prompts:', error);
            throw error;
        }
    }
}

// Export the class
window.ImageExtractor = ImageExtractor; 