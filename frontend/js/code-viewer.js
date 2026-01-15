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

    showAlignmentResults(result) {
        const scorePercent = Math.round(result.alignment_score * 100);
        const scoreClass = scorePercent >= 70 ? 'high' : scorePercent >= 40 ? 'medium' : 'low';

        let html = `
            <div class="alignment-score">
                <div class="score-bar">
                    <div class="score-fill ${scoreClass}" style="width: ${scorePercent}%"></div>
                </div>
                <span class="score-value ${scoreClass}">${scorePercent}%</span>
            </div>
        `;

        if (result.summary) {
            html += `<div class="summary-text">${this.escapeHtml(result.summary)}</div>`;
        }

        if (result.issues && result.issues.length > 0) {
            html += '<div class="alignment-issues">';
            for (const issue of result.issues) {
                html += `
                    <div class="issue ${issue.issue_type}">
                        <div class="issue-type">${issue.issue_type}</div>
                        <div class="issue-description">${this.escapeHtml(issue.description)}</div>
                        ${issue.summary_excerpt ? `<div class="issue-excerpt"><em>"${this.escapeHtml(issue.summary_excerpt)}"</em></div>` : ''}
                    </div>
                `;
            }
            html += '</div>';
        }

        if (result.suggestions && result.suggestions.length > 0) {
            html += `
                <div class="suggestions">
                    <h4>Suggestions</h4>
                    <ul>
                        ${result.suggestions.map(s => `<li>${this.escapeHtml(s)}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        return html;
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
