const API_BASE = 'http://localhost:8000/api';

const api = {
    async uploadPaper(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/papers/upload`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to upload paper');
        }

        return response.json();
    },

    async uploadText(name, content) {
        const response = await fetch(`${API_BASE}/papers/text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name, content }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to upload text');
        }

        return response.json();
    },

    async getPaperContent() {
        const response = await fetch(`${API_BASE}/papers/content`);

        if (!response.ok) {
            if (response.status === 404) return null;
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get paper content');
        }

        return response.json();
    },

    getPdfUrl() {
        return `${API_BASE}/papers/pdf`;
    },

    async loadCodebase(pathOrUrl) {
        const isGithub = pathOrUrl.includes('github.com');
        const body = isGithub
            ? { github_url: pathOrUrl }
            : { path: pathOrUrl };

        const response = await fetch(`${API_BASE}/code/load`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to load codebase');
        }

        return response.json();
    },

    async getCodeFiles() {
        const response = await fetch(`${API_BASE}/code/files`);

        if (!response.ok) {
            if (response.status === 404) return null;
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get code files');
        }

        return response.json();
    },

    async getFileContent(filePath) {
        const response = await fetch(`${API_BASE}/code/file/${encodeURIComponent(filePath)}`);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get file content');
        }

        return response.json();
    },

    async getIndexStatus() {
        const response = await fetch(`${API_BASE}/code/index-status`);
        return response.json();
    },

    // Store the current AbortController for cancellable requests
    currentAnalysisController: null,

    async analyzeHighlight(highlightedText) {
        // Cancel any existing request
        if (this.currentAnalysisController) {
            this.currentAnalysisController.abort();
        }

        // Create new AbortController for this request
        this.currentAnalysisController = new AbortController();

        try {
            const response = await fetch(`${API_BASE}/analysis/highlight`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ highlighted_text: highlightedText }),
                signal: this.currentAnalysisController.signal,
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to analyze highlight');
            }

            return response.json();
        } finally {
            this.currentAnalysisController = null;
        }
    },

    cancelAnalysis() {
        if (this.currentAnalysisController) {
            this.currentAnalysisController.abort();
            this.currentAnalysisController = null;
            return true;
        }
        return false;
    },

    async getStatus() {
        const response = await fetch(`${API_BASE}/status`);
        return response.json();
    },

    // History APIs
    async getPaperHistory() {
        const response = await fetch(`${API_BASE}/papers/history`);
        return response.json();
    },

    async loadPaperFromHistory(paperId) {
        const response = await fetch(`${API_BASE}/papers/history/${paperId}/load`, {
            method: 'POST',
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to load paper from history');
        }

        return response.json();
    },

    async deletePaperFromHistory(paperId) {
        const response = await fetch(`${API_BASE}/papers/history/${paperId}`, {
            method: 'DELETE',
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to delete paper from history');
        }

        return response.json();
    },

    async getCodebaseHistory() {
        const response = await fetch(`${API_BASE}/code/history`);
        return response.json();
    },

    async loadCodebaseFromHistory(codebaseId) {
        const response = await fetch(`${API_BASE}/code/history/${codebaseId}/load`, {
            method: 'POST',
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to load codebase from history');
        }

        return response.json();
    },

    async deleteCodebaseFromHistory(codebaseId) {
        const response = await fetch(`${API_BASE}/code/history/${codebaseId}`, {
            method: 'DELETE',
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to delete codebase from history');
        }

        return response.json();
    },

    // Code-to-PDF Search APIs
    async getCodebaseTree() {
        const response = await fetch(`${API_BASE}/code/tree`);

        if (!response.ok) {
            if (response.status === 404) return null;
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get codebase tree');
        }

        return response.json();
    },

    async getPaperIndexStatus() {
        const response = await fetch(`${API_BASE}/papers/index-status`);
        return response.json();
    },

    // Store controller for code analysis
    currentCodeAnalysisController: null,

    async analyzeCodeHighlight(highlightedCode, filePath = null) {
        // Cancel any existing request
        if (this.currentCodeAnalysisController) {
            this.currentCodeAnalysisController.abort();
        }

        this.currentCodeAnalysisController = new AbortController();

        try {
            const response = await fetch(`${API_BASE}/analysis/code-highlight`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    highlighted_code: highlightedCode,
                    file_path: filePath,
                }),
                signal: this.currentCodeAnalysisController.signal,
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to analyze code highlight');
            }

            return response.json();
        } finally {
            this.currentCodeAnalysisController = null;
        }
    },

    cancelCodeAnalysis() {
        if (this.currentCodeAnalysisController) {
            this.currentCodeAnalysisController.abort();
            this.currentCodeAnalysisController = null;
            return true;
        }
        return false;
    },

    // ========== Audit API ==========

    async startAudit(options = {}) {
        const response = await fetch(`${API_BASE}/audit/run`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                description_source: options.descriptionSource || 'paper',
                text_content: options.textContent || null,
                focus_areas: options.focusAreas || null,
                catastrophic_categories: options.catastrophicCategories || null,
                min_tier: options.minTier || 'tier4',
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to start audit');
        }

        return response.json();
    },

    async getAuditStatus(auditId) {
        const response = await fetch(`${API_BASE}/audit/${auditId}/status`);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get audit status');
        }

        return response.json();
    },

    async getAuditReport(auditId) {
        const response = await fetch(`${API_BASE}/audit/${auditId}/report`);

        if (!response.ok) {
            if (response.status === 202) {
                return { status: 'running' };
            }
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get audit report');
        }

        return response.json();
    },

    async getAuditHistory() {
        const response = await fetch(`${API_BASE}/audit/history`);
        return response.json();
    },

    async deleteAudit(auditId) {
        const response = await fetch(`${API_BASE}/audit/${auditId}`, {
            method: 'DELETE',
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to delete audit');
        }

        return response.json();
    },

    // False Positive Management
    async markFalsePositive(auditId, warningId, reason, appliesTo = 'this_audit') {
        const response = await fetch(`${API_BASE}/audit/${auditId}/false-positive`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                warning_id: warningId,
                reason: reason,
                applies_to: appliesTo,
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to mark false positive');
        }

        return response.json();
    },

    async getFalsePositives(codebaseHash = null) {
        const url = codebaseHash
            ? `${API_BASE}/audit/false-positives?codebase_hash=${codebaseHash}`
            : `${API_BASE}/audit/false-positives`;
        const response = await fetch(url);
        return response.json();
    },

    async removeFalsePositive(fpId) {
        const response = await fetch(`${API_BASE}/audit/false-positive/${fpId}`, {
            method: 'DELETE',
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to remove false positive');
        }

        return response.json();
    },

    async compareAudits(baseId, compareId) {
        const response = await fetch(`${API_BASE}/audit/compare/${baseId}/${compareId}`);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to compare audits');
        }

        return response.json();
    },
};
