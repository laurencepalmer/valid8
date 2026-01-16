class PaperRefsViewer {
    constructor(containerId, onReferenceClick) {
        this.container = document.getElementById(containerId);
        this.onReferenceClick = onReferenceClick;
    }

    showReferences(references, summary = '', threshold = 0) {
        if (!references || references.length === 0) {
            this.container.innerHTML = `
                <p class="placeholder">No relevant paper sections found for the selected code.</p>
            `;
            return;
        }

        // Filter by threshold
        const filteredRefs = references.filter(ref => ref.relevance_score >= threshold);

        if (filteredRefs.length === 0) {
            const thresholdPercent = Math.round(threshold * 100);
            this.container.innerHTML = `
                <p class="placeholder">No paper references meet the ${thresholdPercent}% relevance threshold.</p>
            `;
            return;
        }

        let html = '';

        if (summary) {
            html += `<div class="summary-box"><strong>Summary:</strong> ${this.escapeHtml(summary)}</div>`;
        }

        if (filteredRefs.length < references.length) {
            html += `<div class="filter-info">Showing ${filteredRefs.length} of ${references.length} results</div>`;
        }

        for (let i = 0; i < filteredRefs.length; i++) {
            const ref = filteredRefs[i];
            const relevancePercent = Math.round(ref.relevance_score * 100);
            const pageInfo = ref.page ? `Page ${ref.page}` : 'Unknown page';
            const refId = `paper-ref-${i}`;

            html += `
                <div class="paper-reference" data-ref-id="${refId}" data-page="${ref.page || ''}" data-start="${ref.start_idx}" data-end="${ref.end_idx}">
                    <div class="paper-reference-header" data-toggle="${refId}">
                        <span class="expand-icon"></span>
                        <span class="page-info">${pageInfo}</span>
                        <span class="relevance">${relevancePercent}%</span>
                        <button class="goto-btn" title="Go to this section in paper">Go to</button>
                    </div>
                    <div class="paper-reference-body" id="${refId}">
                        ${ref.explanation ? `<div class="paper-reference-explanation">${this.escapeHtml(ref.explanation)}</div>` : ''}
                        <div class="paper-reference-content">${this.escapeHtml(ref.content)}</div>
                    </div>
                </div>
            `;
        }

        this.container.innerHTML = html;
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Expand/collapse
        this.container.querySelectorAll('.paper-reference-header[data-toggle]').forEach(header => {
            header.addEventListener('click', (e) => {
                if (e.target.classList.contains('goto-btn')) return;

                const refId = header.dataset.toggle;
                const body = document.getElementById(refId);
                const reference = header.closest('.paper-reference');

                if (body && reference) {
                    reference.classList.toggle('expanded');
                }
            });
        });

        // Go to button
        this.container.querySelectorAll('.goto-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const reference = btn.closest('.paper-reference');
                const page = parseInt(reference.dataset.page) || null;
                const startIdx = parseInt(reference.dataset.start);
                const endIdx = parseInt(reference.dataset.end);

                if (this.onReferenceClick) {
                    this.onReferenceClick(page, startIdx, endIdx);
                }
            });
        });
    }

    clear() {
        this.container.innerHTML = '<p class="placeholder">Highlight code text to find relevant paper sections.</p>';
    }

    showLoading() {
        this.container.innerHTML = '<p class="loading">Finding related paper sections...</p>';
    }

    showError(message) {
        this.container.innerHTML = `<p class="error">${this.escapeHtml(message)}</p>`;
    }

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}
