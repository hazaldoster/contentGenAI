document.addEventListener('DOMContentLoaded', function() {
    const brandForm = document.getElementById('brandForm');
    const generateBtn = document.getElementById('generateBtn');
    const loadingPrompts = document.getElementById('loadingPrompts');
    const promptResults = document.getElementById('promptResults');
    const promptContainer = document.querySelector('.prompt-container');
    const videoLoading = document.getElementById('videoLoading');
    
    let brandInput = '';
    let selectedPromptCard = null; // Seçilen prompt kartını takip etmek için değişken
    
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
        
        // Yeni API endpoint'ine istek at
        fetch('/generate-prompt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: brandInput,
                feature_type: 'video'
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            loadingPrompts.classList.add('hidden');
            
            if (data.error) {
                alert('Hata: ' + data.error);
                return;
            }
            
            promptResults.classList.remove('hidden');
            
            // Promptları ekrana ekle
            promptContainer.innerHTML = '';
            
            // Function call bilgisini göster (debug için)
            if (data.function_calls && data.function_calls.length > 0) {
                console.log('Function calls:', data.function_calls);
            }
            
            // Promptları ekle - Her zaman tam 4 prompt göster
            if (data.prompt_data && data.prompt_data.length > 0) {
                // Maksimum 4 prompt göster
                const promptDataToShow = data.prompt_data.slice(0, 4);
                
                // Eğer 4'ten az prompt varsa, eksik olanları boş prompt ile doldur
                while (promptDataToShow.length < 4) {
                    promptDataToShow.push({
                        style: "Belirlenmedi",
                        prompt: "Bu prompt için içerik oluşturulamadı."
                    });
                }
                
                // 4 promptu göster
                promptDataToShow.forEach((promptItem, index) => {
                    const promptCard = document.createElement('div');
                    promptCard.className = 'prompt-card bg-gray-700 p-4 rounded-lg hover:bg-gray-600 transition-colors';
                    
                    // Prompt ID'si oluştur
                    const promptId = `prompt-${index}`;
                    const textareaId = `textarea-${index}`;
                    
                    // Prompt kartı içeriği
                    promptCard.innerHTML = `
                        <div class="mb-2">
                            <h3 class="font-medium text-purple-300">Stil: ${promptItem.style}</h3>
                        </div>
                        <div class="prompt-content" id="${promptId}">
                            <p class="text-gray-300">${promptItem.prompt}</p>
                        </div>
                        <div class="flex justify-center mt-3">
                            <button class="edit-btn bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded transition-colors" data-index="${index}">
                                Düzenle
                            </button>
                        </div>
                        <div class="prompt-edit hidden" id="${textareaId}">
                            <textarea class="w-full bg-gray-800 text-gray-300 p-2 rounded mb-2" rows="5">${promptItem.prompt}</textarea>
                            <div class="flex justify-end space-x-2">
                                <button class="cancel-btn bg-gray-600 hover:bg-gray-500 text-white text-xs px-2 py-1 rounded transition-colors" data-index="${index}">
                                    İptal
                                </button>
                                <button class="save-btn bg-green-600 hover:bg-green-700 text-white text-xs px-2 py-1 rounded transition-colors" data-index="${index}">
                                    Kaydet
                                </button>
                            </div>
                        </div>
                    `;
                    
                    promptContainer.appendChild(promptCard);
                    
                    // Boş promptlar için tıklama olayı ekleme
                    if (index < data.prompt_data.length) {
                        // Prompt kartına tıklama olayı ekle
                        const promptContent = promptCard.querySelector(`#${promptId}`);
                        promptContent.addEventListener('click', function() {
                            // Kartı seçili olarak işaretle
                            selectPromptCard(promptCard);
                            
                            // Seçilen aspect ratio değerini al
                            const aspectRatio = document.querySelector('input[name="aspectRatio"]:checked').value;
                            // Şimdilik sadece seçim yapılsın, video oluşturma işlemi yapılmasın
                        });
                        
                        // Düzenle butonuna tıklama olayı ekle
                        const editBtn = promptCard.querySelector('.edit-btn');
                        editBtn.addEventListener('click', function(e) {
                            e.stopPropagation(); // Kartın tıklama olayını engelle
                            
                            // Düzenleme modunu aç
                            const promptContent = document.getElementById(promptId);
                            const promptEdit = document.getElementById(textareaId);
                            
                            promptContent.classList.add('hidden');
                            promptEdit.classList.remove('hidden');
                        });
                        
                        // İptal butonuna tıklama olayı ekle
                        const cancelBtn = promptCard.querySelector('.cancel-btn');
                        cancelBtn.addEventListener('click', function(e) {
                            e.stopPropagation(); // Kartın tıklama olayını engelle
                            
                            // Düzenleme modunu kapat
                            const promptContent = document.getElementById(promptId);
                            const promptEdit = document.getElementById(textareaId);
                            
                            promptContent.classList.remove('hidden');
                            promptEdit.classList.add('hidden');
                            
                            // Textarea içeriğini orijinal prompt ile değiştir
                            const textarea = promptEdit.querySelector('textarea');
                            textarea.value = promptItem.prompt;
                        });
                        
                        // Kaydet butonuna tıklama olayı ekle
                        const saveBtn = promptCard.querySelector('.save-btn');
                        saveBtn.addEventListener('click', function(e) {
                            e.stopPropagation(); // Kartın tıklama olayını engelle
                            
                            // Yeni prompt değerini al
                            const textarea = document.querySelector(`#${textareaId} textarea`);
                            const newPrompt = textarea.value.trim();
                            
                            if (newPrompt) {
                                // Prompt değerini güncelle
                                promptItem.prompt = newPrompt;
                                
                                // Görünümü güncelle
                                const promptContent = document.getElementById(promptId);
                                promptContent.querySelector('p').textContent = newPrompt;
                                
                                // Düzenleme modunu kapat
                                promptContent.classList.remove('hidden');
                                document.getElementById(textareaId).classList.add('hidden');
                            }
                        });
                    } else {
                        promptCard.classList.add('opacity-50');
                    }
                });
            } else if (data.prompts && data.prompts.length > 0) {
                // Eski format için destek - maksimum 4 prompt göster
                const promptsToShow = data.prompts.slice(0, 4);
                
                // Eğer 4'ten az prompt varsa, eksik olanları boş prompt ile doldur
                while (promptsToShow.length < 4) {
                    promptsToShow.push("Bu prompt için içerik oluşturulamadı.");
                }
                
                promptsToShow.forEach((prompt, index) => {
                    const promptCard = document.createElement('div');
                    promptCard.className = 'prompt-card bg-gray-700 p-4 rounded-lg hover:bg-gray-600 transition-colors';
                    
                    // Prompt ID'si oluştur
                    const promptId = `prompt-${index}`;
                    const textareaId = `textarea-${index}`;
                    
                    // Prompt kartı içeriği
                    promptCard.innerHTML = `
                        <div class="mb-2">
                            <h3 class="font-medium text-purple-300">Prompt ${index + 1}</h3>
                        </div>
                        <div class="prompt-content" id="${promptId}">
                            <p class="text-gray-300">${prompt}</p>
                        </div>
                        <div class="flex justify-center mt-3">
                            <button class="edit-btn bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded transition-colors" data-index="${index}">
                                Düzenle
                            </button>
                        </div>
                        <div class="prompt-edit hidden" id="${textareaId}">
                            <textarea class="w-full bg-gray-800 text-gray-300 p-2 rounded mb-2" rows="5">${prompt}</textarea>
                            <div class="flex justify-end space-x-2">
                                <button class="cancel-btn bg-gray-600 hover:bg-gray-500 text-white text-xs px-2 py-1 rounded transition-colors" data-index="${index}">
                                    İptal
                                </button>
                                <button class="save-btn bg-green-600 hover:bg-green-700 text-white text-xs px-2 py-1 rounded transition-colors" data-index="${index}">
                                    Kaydet
                                </button>
                            </div>
                        </div>
                    `;
                    
                    promptContainer.appendChild(promptCard);
                    
                    // Boş promptlar için tıklama olayı ekleme
                    if (index < data.prompts.length) {
                        // Prompt kartına tıklama olayı ekle
                        const promptContent = promptCard.querySelector(`#${promptId}`);
                        promptContent.addEventListener('click', function() {
                            // Kartı seçili olarak işaretle
                            selectPromptCard(promptCard);
                            
                            // Seçilen aspect ratio değerini al
                            const aspectRatio = document.querySelector('input[name="aspectRatio"]:checked').value;
                            // Şimdilik sadece seçim yapılsın, video oluşturma işlemi yapılmasın
                        });
                        
                        // Düzenle butonuna tıklama olayı ekle
                        const editBtn = promptCard.querySelector('.edit-btn');
                        editBtn.addEventListener('click', function(e) {
                            e.stopPropagation(); // Kartın tıklama olayını engelle
                            
                            // Düzenleme modunu aç
                            const promptContent = document.getElementById(promptId);
                            const promptEdit = document.getElementById(textareaId);
                            
                            promptContent.classList.add('hidden');
                            promptEdit.classList.remove('hidden');
                        });
                        
                        // İptal butonuna tıklama olayı ekle
                        const cancelBtn = promptCard.querySelector('.cancel-btn');
                        cancelBtn.addEventListener('click', function(e) {
                            e.stopPropagation(); // Kartın tıklama olayını engelle
                            
                            // Düzenleme modunu kapat
                            const promptContent = document.getElementById(promptId);
                            const promptEdit = document.getElementById(textareaId);
                            
                            promptContent.classList.remove('hidden');
                            promptEdit.classList.add('hidden');
                            
                            // Textarea içeriğini orijinal prompt ile değiştir
                            const textarea = promptEdit.querySelector('textarea');
                            textarea.value = promptsToShow[index];
                        });
                        
                        // Kaydet butonuna tıklama olayı ekle
                        const saveBtn = promptCard.querySelector('.save-btn');
                        saveBtn.addEventListener('click', function(e) {
                            e.stopPropagation(); // Kartın tıklama olayını engelle
                            
                            // Yeni prompt değerini al
                            const textarea = document.querySelector(`#${textareaId} textarea`);
                            const newPrompt = textarea.value.trim();
                            
                            if (newPrompt) {
                                // Prompt değerini güncelle
                                promptsToShow[index] = newPrompt;
                                
                                // Görünümü güncelle
                                const promptContent = document.getElementById(promptId);
                                promptContent.querySelector('p').textContent = newPrompt;
                                
                                // Düzenleme modunu kapat
                                promptContent.classList.remove('hidden');
                                document.getElementById(textareaId).classList.add('hidden');
                            }
                        });
                    } else {
                        promptCard.classList.add('opacity-50');
                    }
                });
            } else {
                const noPromptMsg = document.createElement('div');
                noPromptMsg.className = 'bg-red-800 p-4 rounded-lg mt-3';
                noPromptMsg.innerHTML = `
                    <h3 class="font-medium text-white mb-2">Hata</h3>
                    <p class="text-gray-200">Prompt oluşturulamadı. Lütfen tekrar deneyin.</p>
                `;
                promptContainer.appendChild(noPromptMsg);
                
                // Form'u tekrar göster
                setTimeout(() => {
                    promptResults.classList.add('hidden');
                }, 3000);
            }
            
            // Aspect ratio seçimini güncelle
            updateSelectedAspectRatio();
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
            } else {
                option.classList.remove('selected');
            }
        });
        
        // Her bir radio input için event listener ekle
        aspectRatioInputs.forEach(input => {
            input.addEventListener('change', function() {
                aspectRatioOptions.forEach(option => {
                    const optionInput = option.querySelector('input');
                    if (optionInput.checked) {
                        option.classList.add('selected');
                    } else {
                        option.classList.remove('selected');
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
}); 