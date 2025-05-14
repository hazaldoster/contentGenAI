// Global variables
let currentImageContainer = null;

// Global functions
function submitImageSelection() {
    if (!currentImageContainer) {
        console.warn('⚠️ Görsel seçilmedi!');
        alert('Lütfen bir görsel seçin');
        return;
    }

    const selectedImageUrl = currentImageContainer.getAttribute('data-url');
    console.log('🖼️ Seçilen görsel:', selectedImageUrl);

    // Generate prompts for the selected image
    fetch('/generate-prompt', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: selectedImageUrl,
            feature_type: 'image'
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('❌ Prompt oluşturma hatası:', data.error);
            throw new Error(data.error);
        }

        // Log prompts to console
        if (data.prompt_data && data.prompt_data.length > 0) {
            console.log('\n📝 Seçilen Görsel için Oluşturulan Promptlar:');
            data.prompt_data.forEach((item, index) => {
                console.log(`\nPrompt ${index + 1}:`);
                console.log('Sahne:', item.scene);
                console.log('Prompt:', item.prompt);
                console.log('------------------------');
            });
        }

        // Close the modal
        closeModal();

        // Add selected class to the image container
        document.querySelectorAll('.selected').forEach(el => el.classList.remove('selected'));
        currentImageContainer.classList.add('selected');
    })
    .catch(error => {
        console.error('❌ Prompt oluşturma hatası:', error);
        alert('Prompt oluşturulurken bir hata oluştu: ' + error.message);
    });
}

function showModal(imageUrl) {
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    
    modalImage.src = imageUrl;
    modal.classList.remove('hidden');
}

function closeModal() {
    const modal = document.getElementById('imageModal');
    modal.classList.add('hidden');
}

document.addEventListener('DOMContentLoaded', function() {
    const brandForm = document.getElementById('brandForm');
    const generateBtn = document.getElementById('generateBtn');
    const loadingPrompts = document.getElementById('loadingPrompts');
    const promptResults = document.getElementById('promptResults');
    const promptContainer = document.querySelector('.prompt-container');
    const videoLoading = document.getElementById('videoLoading');
    
    let brandInput = '';
    let selectedPromptCard = null; // Seçilen prompt kartını takip etmek için değişken
    
    // Auto resize textarea function
    function autoResizeTextarea(textarea) {
        // Set initial height and scroll height
        textarea.style.height = 'auto';
        textarea.style.height = (textarea.scrollHeight) + 'px';
        
        // Add input event listener to resize as user types
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    }
    
    // Prompt kartını seçili olarak işaretleyen fonksiyon
    function selectPromptCard(card) {
        // Önceki seçili kartı temizle
        if (selectedPromptCard) {
            selectedPromptCard.classList.remove('selected');
        }
        
        // Yeni kartı seç
        card.classList.add('selected');
        selectedPromptCard = card;
    }
    
    brandForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const brandInput = document.getElementById('brandInput').value;
        if (!brandInput) {
            alert('Lütfen marka/ürün bilgisi girin');
            return;
        }
        
        // Eski promptları ve video sonuçlarını gizle
        promptResults.classList.add('hidden');
        if (document.getElementById('videoResult')) {
            document.getElementById('videoResult').classList.add('hidden');
        }
        promptContainer.innerHTML = ''; // Eski promptları temizle
        selectedPromptCard = null; // Seçili prompt kartını sıfırla
        
        // Promptları oluşturma işlemi
        loadingPrompts.classList.remove('hidden');
        
        // Seçilen aspect ratio değerini al
        const selectedAspectRatio = document.querySelector('input[name="aspectRatio"]:checked').value;
        
        // Yeni API endpoint'ine istek at
        fetch('/generate-prompt-2', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: brandInput,
                feature_type: 'video',
                aspect_ratio: selectedAspectRatio
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                alert('Hata: ' + data.error);
                return;
            }
            
            // Hide loading state
            loadingPrompts.classList.add('hidden');
            
            // Log function calls if they exist
            if (data.function_calls && data.function_calls.length > 0) {
                console.log('Function calls:', data.function_calls);
            }
            
            // Log prompts to console
            if (data.prompt_data && data.prompt_data.length > 0) {
                console.log('\n📝 Generated Prompts:');
                data.prompt_data.forEach((item, index) => {
                    console.log(`\nPrompt ${index + 1}:`);
                    console.log('Style:', item.style);
                    console.log('Prompt:', item.prompt);
                    console.log('------------------------');
                });
                
                // Show the prompt results container
                promptResults.classList.remove('hidden');
                
                // Clear existing prompts
                promptContainer.innerHTML = '';
                
                // Add prompts to UI
                data.prompt_data.forEach((item, index) => {
                    const promptCard = document.createElement('div');
                    promptCard.className = 'prompt-card p-4 rounded-lg cursor-pointer transition-all relative';
                    promptCard.style.background = '#252525';
                    promptCard.style.textAlign = 'left !important';
                    
                    // Create unique IDs for the prompt content and edit areas
                    const promptId = `prompt-${index}`;
                    const textareaId = `textarea-${index}`;
                    
                    promptCard.innerHTML = `
                        <div class="mb-2" style="text-align: center !important;">
                            <h3 class="font-medium text-[#80CBC4]">${item.style || 'Style ' + (index + 1)}</h3>
                        </div>
                        <div class="prompt-content" id="${promptId}" style="text-align: left !important;">
                            <p class="text-gray-300 phantom-text-regular" style="text-align: left !important;">${item.prompt}</p>
                        </div>
                        <div class="prompt-edit hidden" id="${textareaId}">
                          <textarea class="w-full bg-[#131313] text-gray-300 p-2 rounded mb-2 border border-[#333333]" style="min-height: 200px; height: auto; resize: vertical; display: block; box-sizing: border-box; overflow: auto; white-space: pre-wrap; line-height: 1.5; text-align: left !important;">${item.prompt}</textarea>
                            <div class="flex justify-center space-x-2">
                                <button class="cancel-btn bg-[#131313] hover:bg-[#1a1a1a] text-white text-xs px-2 py-1 rounded transition-colors border border-[#333333]" data-index="${index}">
                                    İptal
                                </button>
                                <button class="save-btn bg-[#131313] hover:bg-[#1a1a1a] text-white text-xs px-2 py-1 rounded transition-colors border border-[#333333]" data-index="${index}">
                                    Kaydet
                                </button>
                            </div>
                        </div>
                        <div class="mt-3 flex justify-center">
                            <button class="edit-btn bg-[#131313] hover:bg-[#1a1a1a] text-white text-xs px-3 py-1 rounded-full transition-colors border border-[#333333]" data-index="${index}">
                                Düzenle
                            </button>
                        </div>
                    `;
                    
                    promptContainer.appendChild(promptCard);
                    
                    // Add click event for the prompt content
                    const promptContent = promptCard.querySelector(`#${promptId}`);
                    promptContent.addEventListener('click', function(e) {
                        // Make sure we don't trigger this when editing
                        if (!e.target.closest('.prompt-edit') && !e.target.closest('.edit-btn')) {
                            selectPromptCard(promptCard);
                        }
                    });
                    
                    // Add edit button functionality
                    const editBtn = promptCard.querySelector('.edit-btn');
                    editBtn.addEventListener('click', function(e) {
                        e.stopPropagation(); // Prevent selecting the card
                        
                        // Show edit mode
                        const promptContent = document.getElementById(promptId);
                        const promptEdit = document.getElementById(textareaId);
                        
                        promptContent.classList.add('hidden');
                        promptEdit.classList.remove('hidden');
                        
                        // Auto-resize the textarea
                        const textarea = promptEdit.querySelector('textarea');
                        autoResizeTextarea(textarea);
                    });
                    
                    // Add cancel button functionality
                    const cancelBtn = promptCard.querySelector('.cancel-btn');
                    cancelBtn.addEventListener('click', function(e) {
                        e.stopPropagation(); // Prevent selecting the card
                        
                        // Hide edit mode
                        const promptContent = document.getElementById(promptId);
                        const promptEdit = document.getElementById(textareaId);
                        
                        promptContent.classList.remove('hidden');
                        promptEdit.classList.add('hidden');
                        
                        // Restore original content
                        const textarea = promptEdit.querySelector('textarea');
                        textarea.value = item.prompt;
                    });
                    
                    // Add save button functionality
                    const saveBtn = promptCard.querySelector('.save-btn');
                    saveBtn.addEventListener('click', function(e) {
                        e.stopPropagation(); // Prevent selecting the card
                        
                        // Get new prompt value
                        const textarea = document.querySelector(`#${textareaId} textarea`);
                        const newPrompt = textarea.value.trim();
                        
                        if (newPrompt) {
                            // Update prompt value
                            item.prompt = newPrompt;
                            
                            // Update displayed content
                            const promptContent = document.getElementById(promptId);
                            promptContent.querySelector('p').textContent = newPrompt;
                            
                            // Hide edit mode
                            promptContent.classList.remove('hidden');
                            document.getElementById(textareaId).classList.add('hidden');
                        }
                    });
                });
            } else if (data.prompts && data.prompts.length > 0) {
                console.log('\n📝 Generated Prompts:');
                data.prompts.forEach((prompt, index) => {
                    console.log(`\nPrompt ${index + 1}:`);
                    console.log(prompt);
                    console.log('------------------------');
                });
                
                // Show the prompt results container
                promptResults.classList.remove('hidden');
                
                // Clear existing prompts
                promptContainer.innerHTML = '';
                
                // Add prompts to UI
                data.prompts.forEach((prompt, index) => {
                    const promptCard = document.createElement('div');
                    promptCard.className = 'prompt-card p-4 rounded-lg cursor-pointer transition-all relative';
                    promptCard.style.background = '#252525';
                    promptCard.style.textAlign = 'left !important';
                    
                    // Create unique IDs for the prompt content and edit areas
                    const promptId = `prompt-simple-${index}`;
                    const textareaId = `textarea-simple-${index}`;
                    
                    promptCard.innerHTML = `
                        <div class="mb-2" style="text-align: center !important;">
                            <h3 class="font-medium text-[#80CBC4]">Style ${index + 1}</h3>
                        </div>
                        <div class="prompt-content" id="${promptId}" style="text-align: left !important;">
                            <p class="text-gray-300 phantom-text-regular" style="text-align: left !important;">${prompt}</p>
                        </div>
                        <div class="prompt-edit hidden" id="${textareaId}">
                            <textarea class="w-full bg-[#131313] text-gray-300 p-2 rounded mb-2 border border-[#333333]" style="min-height: 200px; height: auto; resize: vertical; display: block; box-sizing: border-box; overflow: auto; white-space: pre-wrap; line-height: 1.5; text-align: left !important;">${prompt}</textarea>
                            <div class="flex justify-center space-x-2">
                                <button class="cancel-btn bg-[#131313] hover:bg-[#1a1a1a] text-white text-xs px-2 py-1 rounded transition-colors border border-[#333333]" data-index="${index}">
                                    İptal
                                </button>
                                <button class="save-btn bg-[#131313] hover:bg-[#1a1a1a] text-white text-xs px-2 py-1 rounded transition-colors border border-[#333333]" data-index="${index}">
                                    Kaydet
                                </button>
                            </div>
                        </div>
                        <div class="mt-3 flex justify-center">
                            <button class="edit-btn bg-[#131313] hover:bg-[#1a1a1a] text-white text-xs px-3 py-1 rounded-full transition-colors border border-[#333333]" data-index="${index}">
                                Düzenle
                            </button>
                        </div>
                    `;
                    
                    promptContainer.appendChild(promptCard);
                    
                    // Add click event for the prompt content
                    const promptContent = promptCard.querySelector(`#${promptId}`);
                    promptContent.addEventListener('click', function(e) {
                        // Make sure we don't trigger this when editing
                        if (!e.target.closest('.prompt-edit') && !e.target.closest('.edit-btn')) {
                            selectPromptCard(promptCard);
                        }
                    });
                    
                    // Add edit button functionality  
                    const editBtn = promptCard.querySelector('.edit-btn');
                    editBtn.addEventListener('click', function(e) {
                        e.stopPropagation(); // Prevent selecting the card
                        
                        // Show edit mode
                        const promptContent = document.getElementById(promptId);
                        const promptEdit = document.getElementById(textareaId);
                        
                        promptContent.classList.add('hidden');
                        promptEdit.classList.remove('hidden');
                        
                        // Auto-resize the textarea
                        const textarea = promptEdit.querySelector('textarea');
                        autoResizeTextarea(textarea);
                    });
                    
                    // Add cancel button functionality
                    const cancelBtn = promptCard.querySelector('.cancel-btn');
                    cancelBtn.addEventListener('click', function(e) {
                        e.stopPropagation(); // Prevent selecting the card
                        
                        // Hide edit mode
                        const promptContent = document.getElementById(promptId);
                        const promptEdit = document.getElementById(textareaId);
                        
                        promptContent.classList.remove('hidden');
                        promptEdit.classList.add('hidden');
                        
                        // Restore original content
                        const textarea = promptEdit.querySelector('textarea');
                        textarea.value = prompt;
                    });
                    
                    // Add save button functionality
                    const saveBtn = promptCard.querySelector('.save-btn');
                    saveBtn.addEventListener('click', function(e) {
                        e.stopPropagation(); // Prevent selecting the card
                        
                        // Get new prompt value
                        const textarea = document.querySelector(`#${textareaId} textarea`);
                        const newPrompt = textarea.value.trim();
                        
                        if (newPrompt) {
                            // Update prompt value in the data array
                            data.prompts[index] = newPrompt;
                            
                            // Update displayed content
                            const promptContent = document.getElementById(promptId);
                            promptContent.querySelector('p').textContent = newPrompt;
                            
                            // Hide edit mode
                            promptContent.classList.remove('hidden');
                            document.getElementById(textareaId).classList.add('hidden');
                        }
                    });
                });
            } else {
                console.log('No prompts were generated');
            }
        })
        .catch(error => {
            loadingPrompts.classList.add('hidden');
            console.error('API hatası:', error);
            alert('Bir hata oluştu: ' + error.message);
        });
    });
    
    function createVideo(prompt, brandInput) {
        // Seçilen aspect ratio ve süreyi al
        const selectedAspectRatio = document.querySelector('input[name="aspectRatio"]:checked').value;
        const selectedDuration = document.getElementById('videoDuration').value;
        
        // Video yükleme ekranını göster
        document.getElementById('promptResults').classList.add('hidden');
        document.getElementById('videoLoading').classList.remove('hidden');

        // Form verilerini oluştur
        const formData = new FormData();
        formData.append('prompt', prompt);
        formData.append('brand_input', brandInput);
        formData.append('aspect_ratio', selectedAspectRatio);
        formData.append('duration', selectedDuration);
        formData.append('content_type', 'creative-scene');  // Add content_type for index.html

        // Video oluşturma isteği gönder
        fetch('/generate_video', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            // Video sayfasına yönlendir
            window.location.href = `/video?video_url=${encodeURIComponent(data.video_url)}&prompt=${encodeURIComponent(data.prompt)}&brand=${encodeURIComponent(data.brand_input)}`;
        })
        .catch(error => {
            console.error('Video oluşturma hatası:', error);
            alert('Video oluşturulurken bir hata oluştu: ' + error.message);
            // Hata durumunda yükleme ekranını gizle ve prompt sonuçlarını tekrar göster
            document.getElementById('videoLoading').classList.add('hidden');
            document.getElementById('promptResults').classList.remove('hidden');
        });
    }
    
    // Aspect ratio seçimi için
    function updateSelectedAspectRatio() {
        const aspectRatioInputs = document.querySelectorAll('.aspect-ratio-input');
        const aspectRatioOptions = document.querySelectorAll('.aspect-ratio-option');
        
        aspectRatioOptions.forEach(option => {
            const input = option.querySelector('input');
            if (input.checked) {
                option.classList.add('selected');
                option.style.borderColor = '#FFD166'; // Use yellow brand color for selected border
            } else {
                option.classList.remove('selected');
                option.style.borderColor = '#4b5563'; // Reset to default border-gray-600 color
            }
        });
        
        // Her bir radio input için event listener ekle
        aspectRatioInputs.forEach(input => {
            input.addEventListener('change', function() {
                aspectRatioOptions.forEach(option => {
                    const optionInput = option.querySelector('input');
                    if (optionInput.checked) {
                        option.classList.add('selected');
                        option.style.borderColor = '#FFD166'; // Use yellow brand color for selected border
                    } else {
                        option.classList.remove('selected');
                        option.style.borderColor = '#4b5563'; // Reset to default border-gray-600 color
                    }
                });
            });
        });
    }
    
    // "Seçili Prompt ile Video Oluştur" butonuna tıklama olayı ekle
    document.addEventListener('click', function(e) {
        if (e.target.closest('.create-video-btn')) {
            e.preventDefault();
            
            if (!selectedPromptCard) {
                alert('Lütfen önce bir prompt seçin');
                return;
            }
            
            // Seçilen prompt ve aspect ratio değerlerini al
            const promptContent = selectedPromptCard.querySelector('.prompt-content p');
            const aspectRatio = document.querySelector('input[name="aspectRatio"]:checked').value;
            const brandInputValue = document.getElementById('brandInput').value;
            
            createVideo(promptContent.textContent, brandInputValue);
        }
    });

    // Form submit event listener'ı
    const imageForm = document.getElementById('imageForm');
    if (imageForm) {
        imageForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const brandInput = document.getElementById('brandInput').value;
            console.log('🔍 URL girişi:', brandInput);
            
            if (!brandInput) {
                console.warn('⚠️ URL girilmedi!');
                alert('Lütfen geçerli bir URL girin');
                return;
            }
            
            try {
                console.log('🌐 Görsel çekme işlemi başlatılıyor...');
                imageForm.classList.add('opacity-50', 'pointer-events-none');
                
                const images = await fetchImages(brandInput);
                
                if (images && images.length > 0) {
                    console.log(`✅ Başarıyla ${images.length} görsel çekildi:`);
                    images.forEach((imageUrl, index) => {
                        console.log(`${index + 1}. Görsel: ${imageUrl}`);
                    });

                    // Display images in the grid
                    extractedImagesGrid.innerHTML = '';
                    extractedImagesContainer.classList.remove('hidden');
                    
                    images.forEach((imageUrl, index) => {
                        const imageContainer = document.createElement('div');
                        imageContainer.className = 'relative border-2 border-transparent rounded-lg cursor-pointer hover:border-purple-500 transition-all';
                        imageContainer.setAttribute('data-url', imageUrl);
                        
                        const img = document.createElement('img');
                        img.src = imageUrl;
                        img.alt = `Extracted image ${index + 1}`;
                        img.className = 'w-full h-48 object-cover rounded-lg';
                        
                        const loadingOverlay = document.createElement('div');
                        loadingOverlay.className = 'absolute inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center';
                        loadingOverlay.innerHTML = '<div class="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-purple-500"></div>';
                        imageContainer.appendChild(loadingOverlay);
                        
                        img.onload = function() {
                            loadingOverlay.remove();
                        };
                        
                        img.onerror = function() {
                            loadingOverlay.innerHTML = '<p class="text-red-400 text-sm">Görsel yüklenemedi</p>';
                        };
                        
                        imageContainer.appendChild(img);
                        
                        imageContainer.addEventListener('click', function(e) {
                            if (e.target.tagName === 'IMG') {
                                const imageUrl = this.getAttribute('data-url');
                                currentImageContainer = this;
                                showModal(imageUrl);
                                e.stopPropagation();
                            }
                        });
                        
                        extractedImagesGrid.appendChild(imageContainer);
                    });
                } else {
                    console.warn('⚠️ Hiç görsel bulunamadı');
                    throw new Error('Görsel bulunamadı');
                }

            } catch (error) {
                console.error('❌ İşlem hatası:', {
                    message: error.message,
                    timestamp: new Date().toISOString()
                });
                alert('Hata: ' + error.message);
            } finally {
                imageForm.classList.remove('opacity-50', 'pointer-events-none');
            }
        });
    }
});

async function fetchImages(url) {
    try {
        const imageExtractor = new ImageExtractor();
        const images = await imageExtractor.extractImagesFromUrl(url);
        
        if (!images || images.length === 0) {
            throw new Error('No images found');
        }
        
        return images; // This now returns product_images directly
    } catch (error) {
        console.error('Error fetching images:', error);
        throw error;
    }
}