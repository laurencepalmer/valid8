class CodeViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.codebaseName = null;
    }

    showReferences(references, summary = '', threshold = 0) {
        if (!references || references.length === 0) {
            this.container.innerHTML = `
                <p class="placeholder">No relevant code found for the selected text.</p>
            `;
            return;
        }

        // Filter references by threshold
        const filteredRefs = references.filter(ref => ref.relevance_score >= threshold);

        if (filteredRefs.length === 0) {
            const thresholdPercent = Math.round(threshold * 100);
            this.container.innerHTML = `
                <p class="placeholder">No code references meet the ${thresholdPercent}% relevance threshold. Try lowering the threshold.</p>
            `;
            return;
        }

        let html = '';

        if (summary) {
            html += `<div class="summary-text"><strong>Summary:</strong> ${this.escapeHtml(summary)}</div>`;
        }

        // Show count of filtered results
        if (filteredRefs.length < references.length) {
            html += `<div class="filter-info">Showing ${filteredRefs.length} of ${references.length} results</div>`;
        }

        for (let i = 0; i < filteredRefs.length; i++) {
            const ref = filteredRefs[i];
            const relevancePercent = Math.round(ref.relevance_score * 100);
            const language = this.getLanguageClass(ref.relative_path);
            const refId = `ref-${i}`;

            html += `
                <div class="code-reference" data-ref-id="${refId}">
                    <div class="code-reference-header" data-toggle="${refId}">
                        <span class="expand-icon"></span>
                        <span class="file-path">${this.escapeHtml(ref.relative_path)}</span>
                        <span class="line-info">Lines ${ref.start_line}-${ref.end_line}</span>
                        <span class="relevance">${relevancePercent}%</span>
                    </div>
                    <div class="code-reference-body" id="${refId}">
                        ${ref.explanation ? `<div class="code-reference-explanation">${this.escapeHtml(ref.explanation)}</div>` : ''}
                        <pre><code class="language-${language}">${this.escapeHtml(ref.content)}</code></pre>
                    </div>
                </div>
            `;
        }

        this.container.innerHTML = html;

        // Add click handlers for expanding/collapsing
        this.container.querySelectorAll('.code-reference-header[data-toggle]').forEach(header => {
            header.addEventListener('click', () => {
                const refId = header.dataset.toggle;
                const body = document.getElementById(refId);
                const reference = header.closest('.code-reference');

                if (body && reference) {
                    reference.classList.toggle('expanded');

                    // Highlight code when expanded for the first time
                    if (reference.classList.contains('expanded')) {
                        const codeBlock = body.querySelector('pre code');
                        if (codeBlock && !codeBlock.dataset.highlighted && window.hljs) {
                            hljs.highlightElement(codeBlock);
                            codeBlock.dataset.highlighted = 'true';
                        }
                    }
                }
            });
        });
    }

    clear() {
        this.container.innerHTML = '<p class="placeholder">Load a codebase and highlight paper text to see related code.</p>';

        const nameEl = document.getElementById('codeName');
        if (nameEl) {
            nameEl.textContent = '';
        }
    }

    setCodebaseName(name) {
        this.codebaseName = name;
        const nameEl = document.getElementById('codeName');
        if (nameEl) {
            nameEl.textContent = name;
        }
    }

    getLanguageClass(filePath) {
        const ext = filePath.split('.').pop().toLowerCase();
        const langMap = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'jsx': 'javascript',
            'tsx': 'typescript',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'h': 'c',
            'hpp': 'cpp',
            'go': 'go',
            'rs': 'rust',
            'rb': 'ruby',
            'php': 'php',
            'swift': 'swift',
            'kt': 'kotlin',
            'scala': 'scala',
            'r': 'r',
            'cs': 'csharp',
            'vue': 'vue',
            'svelte': 'svelte',
        };
        return langMap[ext] || 'plaintext';
    }

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}
