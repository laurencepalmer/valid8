class PaperViewer {
    constructor(containerId, onSelect) {
        this.container = document.getElementById(containerId);
        this.onSelect = onSelect;
        this.content = '';
        this.isPdf = false;
        this.pdfDoc = null;
        this.scale = 1.0;
        this.pages = [];
        this.textContent = ''; // Store extracted text for highlighting
        this.setupEventListeners();
        this.setupPdfControls();
    }

    setupEventListeners() {
        this.container.addEventListener('mouseup', () => {
            const selection = window.getSelection();
            const selectedText = selection.toString().trim();

            if (selectedText && selectedText.length >= 10) {
                this.onSelect(selectedText);
            }
        });
    }

    setupPdfControls() {
        const zoomIn = document.getElementById('zoomIn');
        const zoomOut = document.getElementById('zoomOut');

        if (zoomIn) {
            zoomIn.addEventListener('click', () => {
                if (this.scale < 3.0) {
                    this.scale += 0.25;
                    this.updateZoomDisplay();
                    this.rerenderPdf();
                }
            });
        }

        if (zoomOut) {
            zoomOut.addEventListener('click', () => {
                if (this.scale > 0.5) {
                    this.scale -= 0.25;
                    this.updateZoomDisplay();
                    this.rerenderPdf();
                }
            });
        }
    }

    updateZoomDisplay() {
        const zoomLevel = document.getElementById('zoomLevel');
        if (zoomLevel) {
            zoomLevel.textContent = `${Math.round(this.scale * 100)}%`;
        }
    }

    async setPdfUrl(url, name = '') {
        this.isPdf = true;
        this.scale = 1.0;
        this.updateZoomDisplay();

        // Show PDF controls
        const controls = document.getElementById('pdfControls');
        if (controls) {
            controls.classList.remove('hidden');
        }

        // Set paper name
        const nameEl = document.getElementById('paperName');
        if (nameEl) {
            nameEl.textContent = name;
        }

        // Clear container and show loading
        this.container.innerHTML = '<div class="pdf-loading">Loading PDF...</div>';
        this.container.classList.remove('placeholder');
        this.container.classList.add('pdf-viewer');

        try {
            // Configure PDF.js worker
            pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

            // Load the PDF
            const loadingTask = pdfjsLib.getDocument(url);
            this.pdfDoc = await loadingTask.promise;

            // Render all pages
            await this.renderAllPages();
        } catch (error) {
            console.error('Error loading PDF:', error);
            this.container.innerHTML = `<p class="error">Error loading PDF: ${error.message}</p>`;
        }
    }

    async renderAllPages() {
        if (!this.pdfDoc) return;

        // Clear container
        this.container.innerHTML = '';
        this.pages = [];
        this.textContent = '';

        const numPages = this.pdfDoc.numPages;

        for (let pageNum = 1; pageNum <= numPages; pageNum++) {
            const pageContainer = document.createElement('div');
            pageContainer.className = 'pdf-page';
            pageContainer.dataset.pageNum = pageNum;

            const page = await this.pdfDoc.getPage(pageNum);
            const viewport = page.getViewport({ scale: this.scale });

            // Create canvas for rendering
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = viewport.width;
            canvas.height = viewport.height;
            canvas.className = 'pdf-canvas';

            // Create text layer
            const textLayerDiv = document.createElement('div');
            textLayerDiv.className = 'pdf-text-layer';
            textLayerDiv.style.width = `${viewport.width}px`;
            textLayerDiv.style.height = `${viewport.height}px`;

            pageContainer.appendChild(canvas);
            pageContainer.appendChild(textLayerDiv);
            pageContainer.style.width = `${viewport.width}px`;
            pageContainer.style.height = `${viewport.height}px`;

            this.container.appendChild(pageContainer);

            // Render page to canvas
            await page.render({
                canvasContext: context,
                viewport: viewport
            }).promise;

            // Get text content and render text layer
            const textContent = await page.getTextContent();
            this.textContent += this.extractTextFromContent(textContent) + '\n';

            // Render text layer
            await this.renderTextLayer(textLayerDiv, textContent, viewport);

            this.pages.push({
                pageNum,
                canvas,
                textLayerDiv,
                page
            });
        }
    }

    extractTextFromContent(textContent) {
        return textContent.items.map(item => item.str).join(' ');
    }

    async renderTextLayer(container, textContent, viewport) {
        // Clear existing content
        container.innerHTML = '';

        // Render each text item
        textContent.items.forEach(item => {
            if (!item.str.trim()) return;

            const span = document.createElement('span');
            span.textContent = item.str;
            span.className = 'pdf-text-item';

            // Apply transform
            const tx = pdfjsLib.Util.transform(
                viewport.transform,
                item.transform
            );

            const fontSize = Math.sqrt(tx[0] * tx[0] + tx[1] * tx[1]);
            const angle = Math.atan2(tx[1], tx[0]);

            span.style.left = `${tx[4]}px`;
            span.style.top = `${tx[5] - fontSize}px`;
            span.style.fontSize = `${fontSize}px`;
            span.style.fontFamily = item.fontName ? item.fontName : 'sans-serif';

            if (angle !== 0) {
                span.style.transform = `rotate(${angle}rad)`;
            }

            // Set width if available
            if (item.width) {
                span.style.width = `${item.width * this.scale}px`;
            }

            container.appendChild(span);
        });
    }

    async rerenderPdf() {
        if (this.pdfDoc) {
            await this.renderAllPages();
        }
    }

    setContent(content, name = '') {
        this.isPdf = false;
        this.content = content;
        this.textContent = content;
        this.pdfDoc = null;

        // Hide PDF controls
        const controls = document.getElementById('pdfControls');
        if (controls) {
            controls.classList.add('hidden');
        }

        this.container.classList.remove('pdf-viewer');
        this.container.textContent = content;
        this.container.classList.remove('placeholder');

        const nameEl = document.getElementById('paperName');
        if (nameEl) {
            nameEl.textContent = name;
        }
    }

    clear() {
        this.content = '';
        this.textContent = '';
        this.isPdf = false;
        this.pdfDoc = null;
        this.pages = [];

        // Hide PDF controls
        const controls = document.getElementById('pdfControls');
        if (controls) {
            controls.classList.add('hidden');
        }

        this.container.classList.remove('pdf-viewer');
        this.container.innerHTML = '<p class="placeholder">Upload a paper to view its content here. Select text to find related code.</p>';

        const nameEl = document.getElementById('paperName');
        if (nameEl) {
            nameEl.textContent = '';
        }
    }

    highlightText(text) {
        if (!text) return;

        const escapedText = text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const regex = new RegExp(`(${escapedText})`, 'gi');

        if (this.isPdf) {
            // Highlight in PDF text layers
            this.pages.forEach(pageInfo => {
                const spans = pageInfo.textLayerDiv.querySelectorAll('.pdf-text-item');
                spans.forEach(span => {
                    const originalText = span.textContent;
                    if (regex.test(originalText)) {
                        span.innerHTML = originalText.replace(regex, '<mark class="highlight">$1</mark>');
                    }
                });
            });
        } else {
            // Highlight in plain text
            if (!this.content) return;
            this.container.innerHTML = this.content.replace(
                regex,
                '<mark class="highlight">$1</mark>'
            );
        }
    }

    clearHighlights() {
        if (this.isPdf) {
            // Re-render text layers without highlights
            this.pages.forEach(pageInfo => {
                const spans = pageInfo.textLayerDiv.querySelectorAll('.pdf-text-item');
                spans.forEach(span => {
                    const marks = span.querySelectorAll('mark.highlight');
                    marks.forEach(mark => {
                        mark.replaceWith(mark.textContent);
                    });
                });
            });
        } else {
            this.container.textContent = this.content;
        }
    }
}
