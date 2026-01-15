class PaperViewer {
    constructor(containerId, onSelect) {
        this.container = document.getElementById(containerId);
        this.onSelect = onSelect;
        this.content = '';
        this.setupEventListeners();
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

    setContent(content, name = '') {
        this.content = content;
        this.container.textContent = content;
        this.container.classList.remove('placeholder');

        const nameEl = document.getElementById('paperName');
        if (nameEl) {
            nameEl.textContent = name;
        }
    }

    clear() {
        this.content = '';
        this.container.innerHTML = '<p class="placeholder">Upload a paper to view its content here. Select text to find related code.</p>';

        const nameEl = document.getElementById('paperName');
        if (nameEl) {
            nameEl.textContent = '';
        }
    }

    highlightText(text) {
        if (!this.content || !text) return;

        const escapedText = text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const regex = new RegExp(`(${escapedText})`, 'gi');

        this.container.innerHTML = this.content.replace(
            regex,
            '<mark class="highlight">$1</mark>'
        );
    }

    clearHighlights() {
        this.container.textContent = this.content;
    }
}
