document.addEventListener('DOMContentLoaded', () => {
    const paperViewer = new PaperViewer('paperContent', handleTextSelection);
    const codeViewer = new CodeViewer('codeContent');

    let isIndexed = false;

    const elements = {
        sidebar: document.getElementById('sidebar'),
        sidebarToggle: document.getElementById('sidebarToggle'),
        paperFile: document.getElementById('paperFile'),
        pasteTextBtn: document.getElementById('pasteTextBtn'),
        paperStatus: document.getElementById('paperStatus'),
        codePath: document.getElementById('codePath'),
        loadCodeBtn: document.getElementById('loadCodeBtn'),
        codeStatus: document.getElementById('codeStatus'),
        textModal: document.getElementById('textModal'),
        textName: document.getElementById('textName'),
        textContent: document.getElementById('textContent'),
        cancelTextBtn: document.getElementById('cancelTextBtn'),
        submitTextBtn: document.getElementById('submitTextBtn'),
        loadingOverlay: document.getElementById('loadingOverlay'),
        loadingMessage: document.getElementById('loadingMessage'),
        cancelRequestBtn: document.getElementById('cancelRequestBtn'),
        paperHistoryBtn: document.getElementById('paperHistoryBtn'),
        paperHistoryMenu: document.getElementById('paperHistoryMenu'),
        codeHistoryBtn: document.getElementById('codeHistoryBtn'),
        codeHistoryMenu: document.getElementById('codeHistoryMenu'),
        thresholdSlider: document.getElementById('thresholdSlider'),
        thresholdValue: document.getElementById('thresholdValue'),
    };

    // Sidebar toggle functionality
    elements.sidebarToggle.addEventListener('click', () => {
        elements.sidebar.classList.toggle('collapsed');
    });

    // Store last results for threshold filtering
    let lastCodeReferences = [];
    let lastSummary = '';

    function showLoading(message = 'Processing...', showCancel = false) {
        elements.loadingMessage.textContent = message;
        elements.loadingOverlay.classList.remove('hidden');
        if (showCancel) {
            elements.cancelRequestBtn.classList.remove('hidden');
        } else {
            elements.cancelRequestBtn.classList.add('hidden');
        }
    }

    function hideLoading() {
        elements.loadingOverlay.classList.add('hidden');
        elements.cancelRequestBtn.classList.add('hidden');
    }

    function setStatus(element, message, type = '') {
        element.textContent = message;
        element.className = `status ${type}`;
    }

    function formatDate(isoString) {
        const date = new Date(isoString);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);

        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays < 7) return `${diffDays}d ago`;
        return date.toLocaleDateString();
    }

    // Paper History
    async function loadPaperHistory() {
        try {
            const data = await api.getPaperHistory();
            renderPaperHistory(data.papers);
        } catch (error) {
            console.error('Failed to load paper history:', error);
            elements.paperHistoryMenu.innerHTML = '<div class="history-empty">Failed to load history</div>';
        }
    }

    function renderPaperHistory(papers) {
        if (!papers || papers.length === 0) {
            elements.paperHistoryMenu.innerHTML = '<div class="history-empty">No recent papers</div>';
            return;
        }

        elements.paperHistoryMenu.innerHTML = papers.map(paper => `
            <div class="history-item" data-id="${paper.id}">
                <div class="history-item-info">
                    <div class="history-item-name">${escapeHtml(paper.name)}</div>
                    <div class="history-item-meta">${paper.source_type.toUpperCase()} - ${formatDate(paper.uploaded_at)}</div>
                </div>
                <button class="history-item-delete" data-id="${paper.id}" title="Remove from history">x</button>
            </div>
        `).join('');

        // Add click handlers
        elements.paperHistoryMenu.querySelectorAll('.history-item').forEach(item => {
            item.addEventListener('click', async (e) => {
                if (e.target.classList.contains('history-item-delete')) return;
                const paperId = item.dataset.id;
                await loadPaperFromHistory(paperId);
                elements.paperHistoryMenu.classList.add('hidden');
            });
        });

        elements.paperHistoryMenu.querySelectorAll('.history-item-delete').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const paperId = btn.dataset.id;
                await deletePaperFromHistory(paperId);
            });
        });
    }

    async function loadPaperFromHistory(paperId) {
        showLoading('Loading paper...');
        try {
            const result = await api.loadPaperFromHistory(paperId);
            setStatus(elements.paperStatus, `Loaded: ${result.total_length} characters`, 'success');

            if (result.has_pdf) {
                await paperViewer.setPdfUrl(api.getPdfUrl(), result.name);
            } else {
                const paper = await api.getPaperContent();
                if (paper) {
                    paperViewer.setContent(paper.content, paper.name);
                }
            }
        } catch (error) {
            setStatus(elements.paperStatus, error.message, 'error');
        } finally {
            hideLoading();
        }
    }

    async function deletePaperFromHistory(paperId) {
        try {
            await api.deletePaperFromHistory(paperId);
            await loadPaperHistory();
        } catch (error) {
            console.error('Failed to delete paper:', error);
        }
    }

    // Codebase History
    async function loadCodebaseHistory() {
        try {
            const data = await api.getCodebaseHistory();
            renderCodebaseHistory(data.codebases);
        } catch (error) {
            console.error('Failed to load codebase history:', error);
            elements.codeHistoryMenu.innerHTML = '<div class="history-empty">Failed to load history</div>';
        }
    }

    function renderCodebaseHistory(codebases) {
        if (!codebases || codebases.length === 0) {
            elements.codeHistoryMenu.innerHTML = '<div class="history-empty">No recent codebases</div>';
            return;
        }

        elements.codeHistoryMenu.innerHTML = codebases.map(codebase => `
            <div class="history-item" data-id="${codebase.id}">
                <div class="history-item-info">
                    <div class="history-item-name">${escapeHtml(codebase.name)}</div>
                    <div class="history-item-meta">${codebase.file_count} files - ${formatDate(codebase.loaded_at)}</div>
                </div>
                <button class="history-item-delete" data-id="${codebase.id}" title="Remove from history">x</button>
            </div>
        `).join('');

        // Add click handlers
        elements.codeHistoryMenu.querySelectorAll('.history-item').forEach(item => {
            item.addEventListener('click', async (e) => {
                if (e.target.classList.contains('history-item-delete')) return;
                const codebaseId = item.dataset.id;
                await loadCodebaseFromHistory(codebaseId);
                elements.codeHistoryMenu.classList.add('hidden');
            });
        });

        elements.codeHistoryMenu.querySelectorAll('.history-item-delete').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const codebaseId = btn.dataset.id;
                await deleteCodebaseFromHistory(codebaseId);
            });
        });
    }

    async function loadCodebaseFromHistory(codebaseId) {
        showLoading('Loading codebase...');
        setStatus(elements.codeStatus, 'Loading...', 'loading');

        try {
            const result = await api.loadCodebaseFromHistory(codebaseId);
            codeViewer.setCodebaseName(result.name);
            setStatus(
                elements.codeStatus,
                `Loaded ${result.file_count} files (${result.total_lines} lines). Indexing...`,
                'loading'
            );
            pollIndexStatus();
        } catch (error) {
            setStatus(elements.codeStatus, error.message, 'error');
            hideLoading();
        }
    }

    async function deleteCodebaseFromHistory(codebaseId) {
        try {
            await api.deleteCodebaseFromHistory(codebaseId);
            await loadCodebaseHistory();
        } catch (error) {
            console.error('Failed to delete codebase:', error);
        }
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // History dropdown toggle handlers
    elements.paperHistoryBtn.addEventListener('click', async () => {
        const isHidden = elements.paperHistoryMenu.classList.contains('hidden');
        // Close other menus
        elements.codeHistoryMenu.classList.add('hidden');

        if (isHidden) {
            await loadPaperHistory();
            elements.paperHistoryMenu.classList.remove('hidden');
        } else {
            elements.paperHistoryMenu.classList.add('hidden');
        }
    });

    elements.codeHistoryBtn.addEventListener('click', async () => {
        const isHidden = elements.codeHistoryMenu.classList.contains('hidden');
        // Close other menus
        elements.paperHistoryMenu.classList.add('hidden');

        if (isHidden) {
            await loadCodebaseHistory();
            elements.codeHistoryMenu.classList.remove('hidden');
        } else {
            elements.codeHistoryMenu.classList.add('hidden');
        }
    });

    // Close menus when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.history-dropdown')) {
            elements.paperHistoryMenu.classList.add('hidden');
            elements.codeHistoryMenu.classList.add('hidden');
        }
    });

    elements.paperFile.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        showLoading('Uploading paper...');
        try {
            const result = await api.uploadPaper(file);
            setStatus(elements.paperStatus, `Loaded: ${result.total_length} characters`, 'success');

            const paper = await api.getPaperContent();
            if (paper) {
                if (paper.has_pdf) {
                    // Load PDF for visual rendering
                    await paperViewer.setPdfUrl(api.getPdfUrl(), paper.name);
                } else {
                    // Load plain text
                    paperViewer.setContent(paper.content, paper.name);
                }
            }
        } catch (error) {
            setStatus(elements.paperStatus, error.message, 'error');
        } finally {
            hideLoading();
            e.target.value = '';
        }
    });

    elements.pasteTextBtn.addEventListener('click', () => {
        elements.textModal.classList.remove('hidden');
        elements.textContent.focus();
    });

    elements.cancelTextBtn.addEventListener('click', () => {
        elements.textModal.classList.add('hidden');
        elements.textName.value = '';
        elements.textContent.value = '';
    });

    elements.submitTextBtn.addEventListener('click', async () => {
        const content = elements.textContent.value.trim();
        if (!content) {
            alert('Please enter some text');
            return;
        }

        const name = elements.textName.value.trim() || 'Pasted Text';

        showLoading('Processing text...');
        try {
            await api.uploadText(name, content);
            setStatus(elements.paperStatus, `Loaded: ${content.length} characters`, 'success');
            paperViewer.setContent(content, name);

            elements.textModal.classList.add('hidden');
            elements.textName.value = '';
            elements.textContent.value = '';
        } catch (error) {
            setStatus(elements.paperStatus, error.message, 'error');
        } finally {
            hideLoading();
        }
    });

    elements.loadCodeBtn.addEventListener('click', async () => {
        const pathOrUrl = elements.codePath.value.trim();
        if (!pathOrUrl) {
            alert('Please enter a path or GitHub URL');
            return;
        }

        showLoading('Loading codebase...');
        setStatus(elements.codeStatus, 'Loading...', 'loading');

        try {
            const result = await api.loadCodebase(pathOrUrl);
            codeViewer.setCodebaseName(result.name);
            setStatus(
                elements.codeStatus,
                `Loaded ${result.file_count} files (${result.total_lines} lines). Indexing...`,
                'loading'
            );

            pollIndexStatus();
        } catch (error) {
            setStatus(elements.codeStatus, error.message, 'error');
            hideLoading();
        }
    });

    async function pollIndexStatus() {
        const poll = async () => {
            try {
                const status = await api.getIndexStatus();
                if (status.indexed) {
                    isIndexed = true;
                    setStatus(elements.codeStatus, 'Codebase loaded and indexed', 'success');
                    hideLoading();
                } else {
                    // Show progress if available
                    if (status.total > 0) {
                        setStatus(
                            elements.codeStatus,
                            `Indexing... ${status.progress_percent}% (${status.progress}/${status.total} chunks)`,
                            'loading'
                        );
                    }
                    setTimeout(poll, 500);
                }
            } catch (error) {
                setStatus(elements.codeStatus, 'Error checking index status', 'error');
                hideLoading();
            }
        };
        poll();
    }

    async function handleTextSelection(selectedText) {
        if (!isIndexed) {
            alert('Please load and wait for codebase indexing to complete');
            return;
        }

        showLoading('Finding related code...', true);
        try {
            const result = await api.analyzeHighlight(selectedText);
            // Store results for threshold filtering
            lastCodeReferences = result.code_references || [];
            lastSummary = result.summary || '';
            // Apply current threshold filter
            const threshold = parseInt(elements.thresholdSlider.value) / 100;
            codeViewer.showReferences(lastCodeReferences, lastSummary, threshold);
            paperViewer.highlightText(selectedText);
        } catch (error) {
            // Don't show error for cancelled requests
            if (error.name !== 'AbortError') {
                alert(error.message);
            }
        } finally {
            hideLoading();
        }
    }

    // Cancel button handler
    elements.cancelRequestBtn.addEventListener('click', () => {
        api.cancelAnalysis();
        hideLoading();
    });

    // Threshold slider handler
    elements.thresholdSlider.addEventListener('input', () => {
        const value = elements.thresholdSlider.value;
        elements.thresholdValue.textContent = `${value}%`;
        // Re-filter existing results if we have any
        if (lastCodeReferences.length > 0) {
            const threshold = parseInt(value) / 100;
            codeViewer.showReferences(lastCodeReferences, lastSummary, threshold);
        }
    });

    async function checkInitialStatus() {
        try {
            const status = await api.getStatus();

            if (status.paper_loaded) {
                const paper = await api.getPaperContent();
                if (paper) {
                    if (paper.has_pdf) {
                        await paperViewer.setPdfUrl(api.getPdfUrl(), paper.name);
                    } else {
                        paperViewer.setContent(paper.content, paper.name);
                    }
                    setStatus(elements.paperStatus, `Loaded: ${paper.content.length} characters`, 'success');
                }
            }

            if (status.codebase_loaded) {
                codeViewer.setCodebaseName(status.codebase_path);
                if (status.indexed) {
                    isIndexed = true;
                    setStatus(elements.codeStatus, 'Codebase loaded and indexed', 'success');
                } else {
                    setStatus(elements.codeStatus, 'Indexing...', 'loading');
                    pollIndexStatus();
                }
            }
        } catch (error) {
            console.error('Failed to check initial status:', error);
        }
    }

    checkInitialStatus();
});
