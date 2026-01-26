class CodeBrowser {
    constructor(containerId, onCodeSelect) {
        this.container = document.getElementById(containerId);
        this.onCodeSelect = onCodeSelect;
        this.tree = null;
        this.codebaseName = null;
        this.currentFile = null;
        this.selectedFilePath = null;
    }

    async loadTree() {
        try {
            const data = await api.getCodebaseTree();
            if (data && data.tree) {
                this.tree = data.tree;
                this.codebaseName = data.name;
                this.render();
            }
        } catch (error) {
            console.error('Failed to load codebase tree:', error);
            this.container.innerHTML = '<p class="error">Failed to load codebase structure.</p>';
        }
    }

    render() {
        if (!this.tree) {
            this.container.innerHTML = '';
            return;
        }

        this.container.innerHTML = `
            <div class="code-browser-tree">
                ${this.renderTree(this.tree)}
            </div>
            <div class="code-browser-viewer">
                <div class="code-browser-file-header hidden">
                    <span class="file-name"></span>
                    <span class="file-info"></span>
                </div>
                <div class="code-browser-content">
                </div>
            </div>
        `;

        this.setupTreeEventListeners();
    }

    renderTree(node, path = '') {
        let html = '<ul class="tree-list">';

        // Sort entries: directories first, then files
        const entries = Object.entries(node)
            .filter(([key]) => !key.startsWith('_'))
            .sort(([aKey, aVal], [bKey, bVal]) => {
                const aIsDir = aVal._type === 'directory';
                const bIsDir = bVal._type === 'directory';
                if (aIsDir && !bIsDir) return -1;
                if (!aIsDir && bIsDir) return 1;
                return aKey.localeCompare(bKey);
            });

        for (const [name, value] of entries) {
            const itemPath = path ? `${path}/${name}` : name;

            if (value._type === 'directory') {
                html += `
                    <li class="tree-item directory collapsed" data-path="${this.escapeAttr(itemPath)}">
                        <div class="tree-item-header">
                            <span class="tree-icon folder-icon"></span>
                            <span class="tree-name">${this.escapeHtml(name)}</span>
                        </div>
                        <div class="tree-children">
                            ${this.renderTree(value._children, itemPath)}
                        </div>
                    </li>
                `;
            } else {
                html += `
                    <li class="tree-item file" data-path="${this.escapeAttr(value.relative_path)}" data-language="${value.language}">
                        <div class="tree-item-header">
                            <span class="tree-icon file-icon ${value.language}"></span>
                            <span class="tree-name">${this.escapeHtml(name)}</span>
                        </div>
                    </li>
                `;
            }
        }

        html += '</ul>';
        return html;
    }

    setupTreeEventListeners() {
        // Directory expand/collapse
        this.container.querySelectorAll('.tree-item.directory > .tree-item-header').forEach(header => {
            header.addEventListener('click', () => {
                const item = header.closest('.tree-item');
                item.classList.toggle('collapsed');
            });
        });

        // File selection
        this.container.querySelectorAll('.tree-item.file').forEach(item => {
            item.addEventListener('click', async () => {
                // Remove previous selection
                this.container.querySelectorAll('.tree-item.selected').forEach(el => {
                    el.classList.remove('selected');
                });

                item.classList.add('selected');
                const filePath = item.dataset.path;
                await this.loadFile(filePath);
            });
        });
    }

    async loadFile(filePath) {
        const contentArea = this.container.querySelector('.code-browser-content');
        const fileHeader = this.container.querySelector('.code-browser-file-header');

        contentArea.innerHTML = '<p class="loading">Loading file...</p>';

        try {
            const file = await api.getFileContent(filePath);
            this.currentFile = file;
            this.selectedFilePath = filePath;

            // Update header
            fileHeader.classList.remove('hidden');
            fileHeader.querySelector('.file-name').textContent = file.relative_path;
            fileHeader.querySelector('.file-info').textContent = `${file.line_count} lines | ${file.language}`;

            // Render code with syntax highlighting
            const langClass = this.getLanguageClass(file.language);
            contentArea.innerHTML = `
                <pre class="code-browser-pre"><code class="language-${langClass}">${this.escapeHtml(file.content)}</code></pre>
            `;

            // Apply syntax highlighting
            const codeBlock = contentArea.querySelector('code');
            if (window.hljs) {
                hljs.highlightElement(codeBlock);
            }

            // Setup text selection for search
            this.setupSelectionHandler(contentArea);

        } catch (error) {
            contentArea.innerHTML = `<p class="error">Failed to load file: ${error.message}</p>`;
        }
    }

    getLanguageClass(language) {
        const languageMap = {
            'python': 'python',
            'javascript': 'javascript',
            'typescript': 'typescript',
            'java': 'java',
            'go': 'go',
            'rust': 'rust',
            'cpp': 'cpp',
            'c': 'c',
            'csharp': 'csharp',
            'ruby': 'ruby',
            'php': 'php',
            'swift': 'swift',
            'kotlin': 'kotlin',
            'scala': 'scala',
            'r': 'r',
            'shell': 'bash',
            'sql': 'sql',
            'html': 'html',
            'css': 'css',
            'json': 'json',
            'yaml': 'yaml',
            'xml': 'xml',
            'markdown': 'markdown',
        };
        return languageMap[language] || 'plaintext';
    }

    setupSelectionHandler(contentArea) {
        contentArea.addEventListener('mouseup', () => {
            const selection = window.getSelection();
            const selectedText = selection.toString().trim();

            console.log('Code browser mouseup, selected text length:', selectedText.length);

            if (selectedText && selectedText.length >= 10) {
                console.log('Triggering onCodeSelect with:', selectedText.substring(0, 50));
                this.onCodeSelect(selectedText, this.selectedFilePath);
            }
        });
    }

    highlightSelection(text) {
        const contentArea = this.container.querySelector('.code-browser-content');
        if (!contentArea || !this.currentFile) return;

        const codeBlock = contentArea.querySelector('code');
        if (!codeBlock) return;

        // Clear previous highlights
        codeBlock.innerHTML = codeBlock.innerHTML.replace(
            /<mark class="code-highlight">([^<]*)<\/mark>/g,
            '$1'
        );

        // Add new highlight
        const escapedText = text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const regex = new RegExp(`(${escapedText})`, 'g');
        codeBlock.innerHTML = codeBlock.innerHTML.replace(
            regex,
            '<mark class="code-highlight">$1</mark>'
        );
    }

    // Show code search results (when paper text is selected)
    showSearchResults(references, summary = '', threshold = 0) {
        const viewerArea = this.container.querySelector('.code-browser-viewer');
        const contentArea = this.container.querySelector('.code-browser-content');
        const fileHeader = this.container.querySelector('.code-browser-file-header');

        if (!viewerArea || !contentArea) {
            // If tree isn't loaded yet, just show results in the full container
            this.renderSearchResultsFullView(references, summary, threshold);
            return;
        }

        // Hide the file header when showing search results
        if (fileHeader) {
            fileHeader.classList.add('hidden');
        }

        // Filter by threshold
        const filteredRefs = references.filter(ref => ref.relevance_score >= threshold);

        if (!references || references.length === 0) {
            contentArea.innerHTML = '<p class="no-results">No relevant code found for the selected text.</p>';
            return;
        }

        if (filteredRefs.length === 0) {
            const thresholdPercent = Math.round(threshold * 100);
            contentArea.innerHTML = `<p class="no-results">No code references meet the ${thresholdPercent}% relevance threshold.</p>`;
            return;
        }

        let html = '<div class="search-results">';

        if (summary) {
            html += `<div class="summary-text"><strong>Summary:</strong> ${this.escapeHtml(summary)}</div>`;
        }

        if (filteredRefs.length < references.length) {
            html += `<div class="filter-info">Showing ${filteredRefs.length} of ${references.length} results</div>`;
        }

        for (let i = 0; i < filteredRefs.length; i++) {
            const ref = filteredRefs[i];
            const relevancePercent = Math.round(ref.relevance_score * 100);
            const langClass = this.getLanguageClass(ref.relative_path.split('.').pop());
            const refId = `search-ref-${i}`;

            html += `
                <div class="code-reference" data-ref-id="${refId}" data-file-path="${this.escapeAttr(ref.relative_path)}">
                    <div class="code-reference-header" data-toggle="${refId}">
                        <span class="expand-icon"></span>
                        <span class="file-path">${this.escapeHtml(ref.relative_path)}</span>
                        <span class="line-info">Lines ${ref.start_line}-${ref.end_line}</span>
                        <span class="relevance">${relevancePercent}%</span>
                    </div>
                    <div class="code-reference-body" id="${refId}">
                        ${ref.explanation ? `<div class="code-reference-explanation">${this.escapeHtml(ref.explanation)}</div>` : ''}
                        <pre><code class="language-${langClass}">${this.escapeHtml(ref.content)}</code></pre>
                    </div>
                </div>
            `;
        }

        html += '</div>';
        contentArea.innerHTML = html;

        this.setupSearchResultsListeners(contentArea);
    }

    renderSearchResultsFullView(references, summary, threshold) {
        const filteredRefs = references.filter(ref => ref.relevance_score >= threshold);

        if (!references || references.length === 0) {
            this.container.innerHTML = '<p class="no-results">No relevant code found for the selected text.</p>';
            return;
        }

        if (filteredRefs.length === 0) {
            const thresholdPercent = Math.round(threshold * 100);
            this.container.innerHTML = `<p class="no-results">No code references meet the ${thresholdPercent}% relevance threshold.</p>`;
            return;
        }

        let html = '<div class="search-results-full">';

        if (summary) {
            html += `<div class="summary-text"><strong>Summary:</strong> ${this.escapeHtml(summary)}</div>`;
        }

        if (filteredRefs.length < references.length) {
            html += `<div class="filter-info">Showing ${filteredRefs.length} of ${references.length} results</div>`;
        }

        for (let i = 0; i < filteredRefs.length; i++) {
            const ref = filteredRefs[i];
            const relevancePercent = Math.round(ref.relevance_score * 100);
            const langClass = this.getLanguageClass(ref.relative_path.split('.').pop());
            const refId = `search-ref-${i}`;

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
                        <pre><code class="language-${langClass}">${this.escapeHtml(ref.content)}</code></pre>
                    </div>
                </div>
            `;
        }

        html += '</div>';
        this.container.innerHTML = html;

        this.setupSearchResultsListeners(this.container);
    }

    setupSearchResultsListeners(container) {
        // Expand/collapse handlers
        container.querySelectorAll('.code-reference-header[data-toggle]').forEach(header => {
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
        this.tree = null;
        this.codebaseName = null;
        this.currentFile = null;
        this.selectedFilePath = null;
        this.container.innerHTML = '';
    }

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    escapeAttr(text) {
        if (!text) return '';
        return text.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }
}
