document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('csvFile');
    const fileInfo = document.getElementById('fileInfo');
    const dropContent = document.querySelector('.drop-content');
    const fileNameSpan = document.getElementById('fileName');
    const removeFileBtn = document.getElementById('removeFile');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const uploadForm = document.getElementById('uploadForm');
    const btnLoader = document.getElementById('btnLoader');
    const btnText = analyzeBtn.querySelector('span');

    let currentFile = null;

    // --- Drag & Drop Logic ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetFile();
    });

    function handleFile(file) {
        if (file.type !== "text/csv" && !file.name.endsWith('.csv')) {
            showError("Only CSV files are allowed.");
            return;
        }
        
        currentFile = file;
        fileNameSpan.textContent = file.name;
        
        // UI Updates
        dropContent.classList.add('hidden');
        fileInfo.classList.remove('hidden');
        analyzeBtn.disabled = false;
        hideError();
        document.getElementById('resultsContainer').classList.add('hidden');
    }

    function resetFile() {
        currentFile = null;
        fileInput.value = '';
        dropContent.classList.remove('hidden');
        fileInfo.classList.add('hidden');
        analyzeBtn.disabled = true;
        document.getElementById('resultsContainer').classList.add('hidden');
    }

    // --- Form Submission ---
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!currentFile) return;

        // UI Loading State
        analyzeBtn.disabled = true;
        btnText.textContent = "Analyzing...";
        btnLoader.classList.remove('hidden');
        hideError();
        document.getElementById('resultsContainer').classList.add('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Analysis failed');
            }

            displayResults(data);

        } catch (error) {
            showError(error.message);
        } finally {
            // Reset Button State
            analyzeBtn.disabled = false;
            btnText.textContent = "Analyze Sentiment";
            btnLoader.classList.add('hidden');
        }
    });

    function displayResults(data) {
        const resultsContainer = document.getElementById('resultsContainer');
        const posCount = document.getElementById('posCount');
        const neuCount = document.getElementById('neuCount');
        const negCount = document.getElementById('negCount');
        
        const posBar = document.getElementById('posBar');
        const neuBar = document.getElementById('neuBar');
        const negBar = document.getElementById('negBar');

        // Set raw counts
        posCount.textContent = data.positive;
        neuCount.textContent = data.neutral;
        negCount.textContent = data.negative;

        // Calculate percentages for bar widths
        const total = data.positive + data.neutral + data.negative;
        const posPct = total ? (data.positive / total) * 100 : 0;
        const neuPct = total ? (data.neutral / total) * 100 : 0;
        const negPct = total ? (data.negative / total) * 100 : 0;

        // Reveal container
        resultsContainer.classList.remove('hidden');

        // Animate bars (small delay to ensure transition works)
        setTimeout(() => {
            posBar.style.width = `${posPct}%`;
            neuBar.style.width = `${neuPct}%`;
            negBar.style.width = `${negPct}%`;
        }, 100);
    }

    function showError(msg) {
        const errBox = document.getElementById('errorMessage');
        document.getElementById('errorText').textContent = msg;
        errBox.classList.remove('hidden');
    }

    function hideError() {
        document.getElementById('errorMessage').classList.add('hidden');
    }
});