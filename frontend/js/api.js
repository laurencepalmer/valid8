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

    async analyzeHighlight(highlightedText) {
        const response = await fetch(`${API_BASE}/analysis/highlight`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ highlighted_text: highlightedText }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to analyze highlight');
        }

        return response.json();
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
};
