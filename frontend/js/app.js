document.addEventListener('DOMContentLoaded', () => {
    const paperViewer = new PaperViewer('paperContent', handleTextSelection);
    const codeViewer = new CodeViewer('codeContent');

    let currentMode = 'highlight';
    let isIndexed = false;

    const elements = {
        paperFile: document.getElementById('paperFile'),
        pasteTextBtn: document.getElementById('pasteTextBtn'),
        paperStatus: document.getElementById('paperStatus'),
        codePath: document.getElementById('codePath'),
        loadCodeBtn: document.getElementById('loadCodeBtn'),
        codeStatus: document.getElementById('codeStatus'),
        highlightModeBtn: document.getElementById('highlightModeBtn'),
        alignmentModeBtn: document.getElementById('alignmentModeBtn'),
        alignmentPanel: document.getElementById('alignmentPanel'),
        summaryInput: document.getElementById('summaryInput'),
        checkAlignmentBtn: document.getElementById('checkAlignmentBtn'),
        alignmentResults: document.getElementById('alignmentResults'),
        textModal: document.getElementById('textModal'),
        textName: document.getElementById('textName'),
        textContent: document.getElementById('textContent'),
        cancelTextBtn: document.getElementById('cancelTextBtn'),
        submitTextBtn: document.getElementById('submitTextBtn'),
        loadingOverlay: document.getElementById('loadingOverlay'),
        loadingMessage: document.getElementById('loadingMessage'),
    };

    function showLoading(message = 'Processing...') {
        elements.loadingMessage.textContent = message;
        elements.loadingOverlay.classList.remove('hidden');
    }

    function hideLoading() {
        elements.loadingOverlay.classList.add('hidden');
    }

    function setStatus(element, message, type = '') {
        element.textContent = message;
        element.className = `status ${type}`;
    }

    elements.paperFile.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        showLoading('Uploading paper...');
        try {
            const result = await api.uploadPaper(file);
            setStatus(elements.paperStatus, `Loaded: ${result.total_length} characters`, 'success');

            const paper = await api.getPaperContent();
            if (paper) {
                paperViewer.setContent(paper.content, paper.name);
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
        if (currentMode !== 'highlight') return;
        if (!isIndexed) {
            alert('Please load and wait for codebase indexing to complete');
            return;
        }

        showLoading('Finding related code...');
        try {
            const result = await api.analyzeHighlight(selectedText);
            codeViewer.showReferences(result.code_references, result.summary);
            paperViewer.highlightText(selectedText);
        } catch (error) {
            alert(error.message);
        } finally {
            hideLoading();
        }
    }

    elements.highlightModeBtn.addEventListener('click', () => {
        currentMode = 'highlight';
        elements.highlightModeBtn.classList.add('active');
        elements.alignmentModeBtn.classList.remove('active');
        elements.alignmentPanel.classList.add('hidden');
        codeViewer.clear();
        paperViewer.clearHighlights();
    });

    elements.alignmentModeBtn.addEventListener('click', () => {
        currentMode = 'alignment';
        elements.alignmentModeBtn.classList.add('active');
        elements.highlightModeBtn.classList.remove('active');
        elements.alignmentPanel.classList.remove('hidden');
        codeViewer.clear();
        paperViewer.clearHighlights();
    });

    elements.checkAlignmentBtn.addEventListener('click', async () => {
        const summary = elements.summaryInput.value.trim();
        if (!summary) {
            alert('Please enter a summary');
            return;
        }

        if (!isIndexed) {
            alert('Please load and wait for codebase indexing to complete');
            return;
        }

        showLoading('Checking alignment...');
        try {
            const result = await api.checkAlignment(summary);
            elements.alignmentResults.innerHTML = codeViewer.showAlignmentResults(result);
        } catch (error) {
            alert(error.message);
        } finally {
            hideLoading();
        }
    });

    async function checkInitialStatus() {
        try {
            const status = await api.getStatus();

            if (status.paper_loaded) {
                const paper = await api.getPaperContent();
                if (paper) {
                    paperViewer.setContent(paper.content, paper.name);
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
