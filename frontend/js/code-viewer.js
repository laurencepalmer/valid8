class CodeViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
    }

    showReferences(references, summary = '') {
        if (!references || references.length === 0) {
            this.container.innerHTML = `
                <p class="placeholder">No relevant code found for the selected text.</p>
            `;
            return;
        }

        let html = '';

        if (summary) {
            html += `<div class="summary-text"><strong>Summary:</strong> ${this.escapeHtml(summary)}</div>`;
        }

        for (const ref of references) {
            const relevancePercent = Math.round(ref.relevance_score * 100);
            const language = this.getLanguageClass(ref.relative_path);

            html += `
                <div class="code-reference">
                    <div class="code-reference-header">
                        <span class="file-path">${this.escapeHtml(ref.relative_path)}</span>
                        <span class="relevance">${relevancePercent}% relevant</span>
                    </div>
                    ${ref.explanation ? `<div class="code-reference-explanation">${this.escapeHtml(ref.explanation)}</div>` : ''}
                    <div class="line-numbers">Lines ${ref.start_line} - ${ref.end_line}</div>
                    <pre><code class="language-${language}">${this.escapeHtml(ref.content)}</code></pre>
                </div>
            `;
        }

        this.container.innerHTML = html;

        this.container.querySelectorAll('pre code').forEach((block) => {
            if (window.hljs) {
                hljs.highlightElement(block);
            }
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
