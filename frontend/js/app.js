document.addEventListener('DOMContentLoaded', () => {
    // Paper viewer - handles PDF/text display and text selection
    const paperViewer = new PaperViewer('paperContent', handleTextSelection);

    // Code browser - handles file browsing AND code search results display
    const codeBrowser = new CodeBrowser('codeContent', handleCodeSelection);

    let isIndexed = false;
    let isPaperIndexed = false;

    // Query history storage (in memory, persists during session)
    let queryHistory = [];
    const MAX_QUERY_HISTORY = 20;

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
        codebaseName: document.getElementById('codebaseName'),
        clearPaperHighlights: document.getElementById('clearPaperHighlights'),
        clearCodeResults: document.getElementById('clearCodeResults'),
        queryHistoryBtn: document.getElementById('queryHistoryBtn'),
        queryHistoryMenu: document.getElementById('queryHistoryMenu'),
    };

    // Sidebar toggle functionality
    elements.sidebarToggle.addEventListener('click', () => {
        elements.sidebar.classList.toggle('collapsed');
    });

    // Store last results for threshold filtering
    let lastCodeReferences = [];
    let lastSummary = '';

    // Handle code text selection - searches ONLY the paper for relevant sections
    async function handleCodeSelection(selectedCode, filePath) {
        console.log('handleCodeSelection called:', { selectedCode: selectedCode.substring(0, 50), filePath });
        console.log('isPaperIndexed:', isPaperIndexed);

        if (!isPaperIndexed) {
            alert('Please load a paper and wait for indexing to complete');
            return;
        }

        showLoading('Finding related paper sections...', true);

        try {
            console.log('Calling API analyzeCodeHighlight...');
            const result = await api.analyzeCodeHighlight(selectedCode, filePath);
            console.log('API result:', result);
            const references = result.paper_references || [];
            console.log('References count:', references.length);

            // Add to query history
            addToQueryHistory('code-to-paper', selectedCode, references.length);

            if (references.length > 0) {
                // Highlight the top matching text directly in the PDF
                const topRef = references[0];
                console.log('Top reference:', topRef);
                console.log('Top reference page:', topRef.page);

                paperViewer.clearHighlights();
                // Pass the target page to focus highlighting there
                paperViewer.highlightText(topRef.content, topRef.page);

                // Show clear button for paper highlights
                elements.clearPaperHighlights.classList.remove('hidden');
                elements.queryHistoryBtn.classList.remove('hidden');
            } else {
                alert('No related paper sections found for the selected code.');
            }

            // Do NOT highlight the code - only show paper results
            // The user selected code to find paper sections, not to highlight their code
        } catch (error) {
            console.error('handleCodeSelection error:', error);
            if (error.name !== 'AbortError') {
                alert(error.message);
            }
        } finally {
            hideLoading();
        }
    }

    // Handle paper text selection - searches ONLY the codebase for relevant code
    async function handleTextSelection(selectedText) {
        if (!isIndexed) {
            alert('Please load and wait for codebase indexing to complete');
            return;
        }

        showLoading('Finding related code...', true);
        try {
            // Note: This ONLY searches the codebase, not the paper
            const result = await api.analyzeHighlight(selectedText);
            lastCodeReferences = result.code_references || [];
            lastSummary = result.summary || '';

            // Add to query history
            addToQueryHistory('paper-to-code', selectedText, lastCodeReferences.length);

            const threshold = parseInt(elements.thresholdSlider.value) / 100;
            codeBrowser.showSearchResults(lastCodeReferences, lastSummary, threshold);

            // Do NOT highlight the paper - only show code results
            // The user selected paper text to find code, not to highlight the paper

            // Show clear buttons
            if (lastCodeReferences.length > 0) {
                elements.clearCodeResults.classList.remove('hidden');
            }
            elements.queryHistoryBtn.classList.remove('hidden');
        } catch (error) {
            if (error.name !== 'AbortError') {
                alert(error.message);
            }
        } finally {
            hideLoading();
        }
    }

    // Poll paper index status after paper upload
    async function pollPaperIndexStatus() {
        const poll = async () => {
            try {
                const status = await api.getPaperIndexStatus();
                if (status.indexed) {
                    isPaperIndexed = true;
                } else {
                    setTimeout(poll, 500);
                }
            } catch (error) {
                console.error('Error checking paper index status:', error);
            }
        };
        poll();
    }

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

    function setCodebaseName(name) {
        if (elements.codebaseName) {
            elements.codebaseName.textContent = name;
        }
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
        isPaperIndexed = false;
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

            pollPaperIndexStatus();
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
            setCodebaseName(result.name);
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
        isPaperIndexed = false;
        try {
            const result = await api.uploadPaper(file);
            setStatus(elements.paperStatus, `Loaded: ${result.total_length} characters`, 'success');

            const paper = await api.getPaperContent();
            if (paper) {
                if (paper.has_pdf) {
                    await paperViewer.setPdfUrl(api.getPdfUrl(), paper.name);
                } else {
                    paperViewer.setContent(paper.content, paper.name);
                }
            }

            pollPaperIndexStatus();
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
        isPaperIndexed = false;
        try {
            await api.uploadText(name, content);
            setStatus(elements.paperStatus, `Loaded: ${content.length} characters`, 'success');
            paperViewer.setContent(content, name);

            elements.textModal.classList.add('hidden');
            elements.textName.value = '';
            elements.textContent.value = '';

            pollPaperIndexStatus();
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
            setCodebaseName(result.name);
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
                    // Load the code browser tree now that indexing is complete
                    codeBrowser.loadTree();
                } else {
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

    // Cancel button handler
    elements.cancelRequestBtn.addEventListener('click', () => {
        api.cancelAnalysis();
        api.cancelCodeAnalysis();
        hideLoading();
    });

    // Threshold slider handler - re-filter code search results
    elements.thresholdSlider.addEventListener('input', () => {
        const value = elements.thresholdSlider.value;
        elements.thresholdValue.textContent = `${value}%`;
        if (lastCodeReferences.length > 0) {
            const threshold = parseInt(value) / 100;
            codeBrowser.showSearchResults(lastCodeReferences, lastSummary, threshold);
        }
    });

    // Clear paper highlights button handler
    elements.clearPaperHighlights.addEventListener('click', () => {
        paperViewer.clearHighlights();
        elements.clearPaperHighlights.classList.add('hidden');
    });

    // Clear code results button handler
    elements.clearCodeResults.addEventListener('click', () => {
        lastCodeReferences = [];
        lastSummary = '';
        codeBrowser.loadTree(); // Reload the tree view
        elements.clearCodeResults.classList.add('hidden');
    });

    // Query history functions
    function addToQueryHistory(type, text, resultCount) {
        const entry = {
            type,
            text: text.substring(0, 200), // Truncate for storage
            fullText: text,
            resultCount,
            timestamp: new Date().toISOString()
        };

        // Add to beginning
        queryHistory.unshift(entry);

        // Limit size
        if (queryHistory.length > MAX_QUERY_HISTORY) {
            queryHistory = queryHistory.slice(0, MAX_QUERY_HISTORY);
        }

        // Save to localStorage
        try {
            localStorage.setItem('valid8_query_history', JSON.stringify(queryHistory));
        } catch (e) {
            console.warn('Could not save query history to localStorage:', e);
        }
    }

    function loadQueryHistory() {
        try {
            const saved = localStorage.getItem('valid8_query_history');
            if (saved) {
                queryHistory = JSON.parse(saved);
            }
        } catch (e) {
            console.warn('Could not load query history from localStorage:', e);
            queryHistory = [];
        }
    }

    function renderQueryHistory() {
        if (queryHistory.length === 0) {
            elements.queryHistoryMenu.innerHTML = '<div class="query-history-empty">No queries yet</div>';
            return;
        }

        let html = '';
        queryHistory.forEach((entry, index) => {
            const typeLabel = entry.type === 'paper-to-code' ? 'Paper → Code' : 'Code → Paper';
            const timeAgo = formatTimeAgo(entry.timestamp);
            const preview = entry.text.substring(0, 60) + (entry.text.length > 60 ? '...' : '');

            html += `
                <div class="query-history-item" data-index="${index}">
                    <div class="query-history-item-type">${typeLabel} (${entry.resultCount} results)</div>
                    <div class="query-history-item-text">${escapeHtml(preview)}</div>
                    <div class="query-history-item-time">${timeAgo}</div>
                </div>
            `;
        });

        html += `
            <div class="query-history-clear">
                <button id="clearQueryHistory">Clear History</button>
            </div>
        `;

        elements.queryHistoryMenu.innerHTML = html;

        // Add click handlers for history items
        elements.queryHistoryMenu.querySelectorAll('.query-history-item').forEach(item => {
            item.addEventListener('click', () => {
                const index = parseInt(item.dataset.index);
                const entry = queryHistory[index];
                if (entry) {
                    rerunQuery(entry);
                }
                elements.queryHistoryMenu.classList.add('hidden');
            });
        });

        // Add clear history handler
        const clearBtn = elements.queryHistoryMenu.querySelector('#clearQueryHistory');
        if (clearBtn) {
            clearBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                queryHistory = [];
                try {
                    localStorage.removeItem('valid8_query_history');
                } catch (e) {}
                elements.queryHistoryMenu.classList.add('hidden');
            });
        }
    }

    async function rerunQuery(entry) {
        if (entry.type === 'paper-to-code') {
            if (!isIndexed) {
                alert('Please load a codebase first');
                return;
            }
            await handleTextSelection(entry.fullText);
        } else if (entry.type === 'code-to-paper') {
            if (!isPaperIndexed) {
                alert('Please load a paper first');
                return;
            }
            await handleCodeSelection(entry.fullText, null);
        }
    }

    function formatTimeAgo(isoString) {
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

    // Query history button handler
    elements.queryHistoryBtn.addEventListener('click', () => {
        const isHidden = elements.queryHistoryMenu.classList.contains('hidden');
        // Close other dropdowns
        elements.paperHistoryMenu.classList.add('hidden');
        elements.codeHistoryMenu.classList.add('hidden');

        if (isHidden) {
            renderQueryHistory();
            elements.queryHistoryMenu.classList.remove('hidden');
        } else {
            elements.queryHistoryMenu.classList.add('hidden');
        }
    });

    // Close query history menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.query-history-dropdown')) {
            elements.queryHistoryMenu.classList.add('hidden');
        }
    });

    // Load query history on startup
    loadQueryHistory();
    if (queryHistory.length > 0) {
        elements.queryHistoryBtn.classList.remove('hidden');
    }

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

                const paperIndexStatus = await api.getPaperIndexStatus();
                if (paperIndexStatus.indexed) {
                    isPaperIndexed = true;
                } else {
                    pollPaperIndexStatus();
                }
            }

            if (status.codebase_loaded) {
                setCodebaseName(status.codebase_path);
                if (status.indexed) {
                    isIndexed = true;
                    setStatus(elements.codeStatus, 'Codebase loaded and indexed', 'success');
                    // Load the code browser tree
                    codeBrowser.loadTree();
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
