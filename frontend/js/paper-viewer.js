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
        this.zoomTimeout = null; // Debounce timer for zoom
        // Store current highlight state for persistence across zoom
        this.currentHighlight = null; // { text, targetPage }
        this.setupEventListeners();
        this.setupPdfControls();
        this.setupPinchZoom();
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

        console.log('setupPdfControls:', { zoomIn: !!zoomIn, zoomOut: !!zoomOut });

        if (zoomIn) {
            zoomIn.addEventListener('click', (e) => {
                e.stopPropagation();
                console.log('Zoom in clicked, current scale:', this.scale);
                this.zoomTo(this.scale + 0.25);
            });
        }

        if (zoomOut) {
            zoomOut.addEventListener('click', (e) => {
                e.stopPropagation();
                console.log('Zoom out clicked, current scale:', this.scale);
                this.zoomTo(this.scale - 0.25);
            });
        }
    }

    setupPinchZoom() {
        // Handle trackpad pinch-to-zoom (wheel event with ctrlKey)
        this.container.addEventListener('wheel', (e) => {
            if (!this.isPdf || !this.pdfDoc) return;

            // Trackpad pinch gestures are reported as wheel events with ctrlKey
            if (e.ctrlKey) {
                e.preventDefault();

                // Calculate zoom delta (negative deltaY = zoom in)
                const zoomDelta = -e.deltaY * 0.01;
                const newScale = Math.max(0.25, Math.min(5.0, this.scale + zoomDelta));

                if (newScale !== this.scale) {
                    this.scale = newScale;
                    this.updateZoomDisplay();

                    // Debounce the re-render for smoother zooming
                    if (this.zoomTimeout) {
                        clearTimeout(this.zoomTimeout);
                    }
                    this.zoomTimeout = setTimeout(() => {
                        this.rerenderPdf();
                    }, 100);
                }
            }
        }, { passive: false });

        // Handle touch pinch zoom for mobile/tablet
        let initialDistance = 0;
        let initialScale = 1;

        this.container.addEventListener('touchstart', (e) => {
            if (!this.isPdf || !this.pdfDoc) return;
            if (e.touches.length === 2) {
                initialDistance = this.getTouchDistance(e.touches);
                initialScale = this.scale;
            }
        }, { passive: true });

        this.container.addEventListener('touchmove', (e) => {
            if (!this.isPdf || !this.pdfDoc) return;
            if (e.touches.length === 2) {
                e.preventDefault();
                const currentDistance = this.getTouchDistance(e.touches);
                const scaleFactor = currentDistance / initialDistance;
                const newScale = Math.max(0.25, Math.min(5.0, initialScale * scaleFactor));

                if (newScale !== this.scale) {
                    this.scale = newScale;
                    this.updateZoomDisplay();

                    if (this.zoomTimeout) {
                        clearTimeout(this.zoomTimeout);
                    }
                    this.zoomTimeout = setTimeout(() => {
                        this.rerenderPdf();
                    }, 100);
                }
            }
        }, { passive: false });
    }

    getTouchDistance(touches) {
        const dx = touches[0].clientX - touches[1].clientX;
        const dy = touches[0].clientY - touches[1].clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    zoomTo(newScale) {
        newScale = Math.max(0.25, Math.min(5.0, newScale));
        console.log('zoomTo called:', { newScale, currentScale: this.scale, isPdf: this.isPdf, hasPdfDoc: !!this.pdfDoc });
        if (newScale !== this.scale && this.isPdf && this.pdfDoc) {
            this.scale = newScale;
            this.updateZoomDisplay();
            this.rerenderPdf();
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
        this.currentHighlight = null; // Clear any previous highlight
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
            // Re-apply highlights after re-render (without scrolling)
            if (this.currentHighlight) {
                this.applyHighlight(this.currentHighlight.text, this.currentHighlight.targetPage, false);
            }
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
        this.currentHighlight = null;

        // Hide PDF controls
        const controls = document.getElementById('pdfControls');
        if (controls) {
            controls.classList.add('hidden');
        }

        this.container.classList.remove('pdf-viewer');
        this.container.innerHTML = '';

        const nameEl = document.getElementById('paperName');
        if (nameEl) {
            nameEl.textContent = '';
        }
    }

    highlightText(text, targetPage = null) {
        if (!text) return;

        if (this.isPdf) {
            // Store highlight state for persistence across zoom
            this.currentHighlight = { text, targetPage };
            // Apply highlight with scrolling
            this.applyHighlight(text, targetPage, true);
        } else {
            // Highlight in plain text - can do exact match
            if (!this.content) return;
            const escapedText = text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`(${escapedText})`, 'gi');
            this.container.innerHTML = this.content.replace(
                regex,
                '<mark class="highlight">$1</mark>'
            );
        }
    }

    // Apply highlight to PDF text layer (reusable for zoom persistence)
    applyHighlight(text, targetPage, shouldScroll = true) {
        // Get the page to highlight on
        const pagesToSearch = targetPage
            ? this.pages.filter(p => p.pageNum === targetPage)
            : this.pages;

        if (pagesToSearch.length === 0) return;

        // Build a word set from the content for matching
        const contentWords = this.extractContentWords(text);
        if (contentWords.size === 0) return;

        // Score each span by how many content words it contains
        let scoredSpans = [];

        pagesToSearch.forEach(pageInfo => {
            const spans = pageInfo.textLayerDiv.querySelectorAll('.pdf-text-item');
            spans.forEach((span, index) => {
                const spanText = span.textContent.toLowerCase();
                const spanWords = spanText.split(/\s+/).map(w => w.replace(/[^\w]/g, ''));

                let score = 0;
                spanWords.forEach(word => {
                    if (word.length >= 4 && contentWords.has(word)) {
                        score++;
                    }
                });

                if (score > 0) {
                    scoredSpans.push({
                        span,
                        index,
                        pageNum: pageInfo.pageNum,
                        score,
                        top: parseFloat(span.style.top) || 0
                    });
                }
            });
        });

        if (scoredSpans.length === 0) return;

        // Sort by vertical position to find clusters
        scoredSpans.sort((a, b) => a.top - b.top);

        // Find the best cluster of consecutive spans (likely a paragraph)
        let bestCluster = this.findBestCluster(scoredSpans);

        // Highlight all spans in the best cluster
        bestCluster.forEach(({ span }) => {
            span.classList.add('section-highlight');
        });

        // Scroll to the first highlighted span (only on initial highlight, not on zoom)
        if (shouldScroll && bestCluster.length > 0) {
            this.scrollToPage(bestCluster[0].pageNum);
            // Also scroll the span into view within the page
            setTimeout(() => {
                bestCluster[0].span.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 300);
        }
    }

    // Extract significant words from content for matching
    extractContentWords(text) {
        const stopWords = new Set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'that', 'which', 'who', 'whom', 'this', 'these', 'those', 'it', 'its',
            'as', 'if', 'then', 'than', 'so', 'such', 'both', 'each', 'all', 'any',
            'some', 'no', 'not', 'only', 'same', 'also', 'very', 'just', 'even',
            'because', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'into', 'over', 'under', 'again', 'further', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'what', 'while', 'although',
            'using', 'based', 'given', 'shown', 'used', 'since', 'thus', 'hence',
            'paper', 'method', 'approach', 'results', 'figure', 'table', 'section'
        ]);

        const words = text.toLowerCase().split(/\s+/).map(w => w.replace(/[^\w]/g, ''));
        const wordSet = new Set();

        words.forEach(word => {
            if (word.length >= 4 && !stopWords.has(word)) {
                wordSet.add(word);
            }
        });

        return wordSet;
    }

    // Find the best cluster of spans that likely form a paragraph
    findBestCluster(scoredSpans) {
        if (scoredSpans.length === 0) return [];
        if (scoredSpans.length <= 3) return scoredSpans;

        let bestCluster = [];
        let bestScore = 0;

        // Use sliding window to find best cluster
        const windowSize = Math.min(10, scoredSpans.length);

        for (let i = 0; i <= scoredSpans.length - windowSize; i++) {
            const window = scoredSpans.slice(i, i + windowSize);

            // Check if spans are vertically close (within ~100px of each other)
            const topMin = window[0].top;
            const topMax = window[window.length - 1].top;
            const verticalSpread = topMax - topMin;

            // Calculate total score for this window
            const totalScore = window.reduce((sum, s) => sum + s.score, 0);

            // Prefer tighter clusters with higher scores
            const clusterScore = totalScore / (1 + verticalSpread / 100);

            if (clusterScore > bestScore) {
                bestScore = clusterScore;
                bestCluster = window;
            }
        }

        // Also include nearby spans with any matches
        if (bestCluster.length > 0) {
            const clusterTop = bestCluster[0].top;
            const clusterBottom = bestCluster[bestCluster.length - 1].top;
            const margin = 30; // Include spans within 30px

            scoredSpans.forEach(span => {
                if (!bestCluster.includes(span) &&
                    span.top >= clusterTop - margin &&
                    span.top <= clusterBottom + margin) {
                    bestCluster.push(span);
                }
            });

            // Re-sort by position
            bestCluster.sort((a, b) => a.top - b.top);
        }

        return bestCluster;
    }

    clearHighlights() {
        // Clear stored highlight state
        this.currentHighlight = null;

        if (this.isPdf) {
            // Re-render text layers without highlights
            this.pages.forEach(pageInfo => {
                const spans = pageInfo.textLayerDiv.querySelectorAll('.pdf-text-item');
                spans.forEach(span => {
                    // Remove section highlight class
                    span.classList.remove('section-highlight');

                    // Remove mark elements
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

    scrollToPage(pageNum) {
        if (!this.isPdf || !this.pdfDoc) return;

        const pageContainer = this.container.querySelector(`.pdf-page[data-page-num="${pageNum}"]`);
        if (pageContainer) {
            pageContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }
}
