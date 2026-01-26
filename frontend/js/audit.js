/**
 * Audit functionality for Valid8
 * Handles audit execution, result display, and false positive management
 */

class AuditManager {
    constructor() {
        this.currentAuditId = null;
        this.currentReport = null;
        this.pollInterval = null;

        this.elements = {
            runAuditBtn: document.getElementById('runAuditBtn'),
            auditHistoryBtn: document.getElementById('auditHistoryBtn'),
            auditHistoryMenu: document.getElementById('auditHistoryMenu'),
            auditStatus: document.getElementById('auditStatus'),
            auditModal: document.getElementById('auditModal'),
            auditModalBody: document.getElementById('auditModalBody'),
            closeAuditModal: document.getElementById('closeAuditModal'),
            fpModal: document.getElementById('fpModal'),
            fpWarningDescription: document.getElementById('fpWarningDescription'),
            fpReason: document.getElementById('fpReason'),
            fpScope: document.getElementById('fpScope'),
            cancelFpBtn: document.getElementById('cancelFpBtn'),
            submitFpBtn: document.getElementById('submitFpBtn'),
        };

        this.pendingFpWarningId = null;
        this.bindEvents();
    }

    bindEvents() {
        // Run audit button
        this.elements.runAuditBtn.addEventListener('click', () => this.startAudit());

        // Audit history
        this.elements.auditHistoryBtn.addEventListener('click', () => this.toggleHistoryMenu());

        // Close audit modal
        this.elements.closeAuditModal.addEventListener('click', () => this.closeModal());

        // Close modal on outside click
        this.elements.auditModal.addEventListener('click', (e) => {
            if (e.target === this.elements.auditModal) {
                this.closeModal();
            }
        });

        // False positive modal
        this.elements.cancelFpBtn.addEventListener('click', () => this.closeFpModal());
        this.elements.submitFpBtn.addEventListener('click', () => this.submitFalsePositive());

        // Close history menu on outside click
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.audit-section')) {
                this.elements.auditHistoryMenu.classList.add('hidden');
            }
        });
    }

    async startAudit() {
        this.setStatus('Starting audit...', 'loading');

        try {
            const result = await api.startAudit({
                descriptionSource: 'paper',
                minTier: 'tier4',
            });

            this.currentAuditId = result.audit_id;
            this.setStatus('Audit running...', 'loading');
            this.pollAuditStatus();
        } catch (error) {
            this.setStatus(error.message, 'error');
        }
    }

    async pollAuditStatus() {
        if (!this.currentAuditId) return;

        try {
            const status = await api.getAuditStatus(this.currentAuditId);

            if (status.status === 'completed') {
                this.setStatus('Audit complete', 'success');
                await this.showAuditReport(this.currentAuditId);
            } else if (status.status === 'failed') {
                this.setStatus('Audit failed', 'error');
            } else {
                // Still running
                const progressPct = Math.round(status.progress * 100);
                this.setStatus(`${status.current_step} (${progressPct}%)`, 'loading');
                setTimeout(() => this.pollAuditStatus(), 1000);
            }
        } catch (error) {
            this.setStatus('Error checking status', 'error');
        }
    }

    async showAuditReport(auditId) {
        try {
            const report = await api.getAuditReport(auditId);
            if (report.status === 'running') {
                // Still running, poll again
                setTimeout(() => this.showAuditReport(auditId), 1000);
                return;
            }

            this.currentReport = report;
            this.currentAuditId = auditId;
            this.renderReport(report);
            this.openModal();
        } catch (error) {
            this.setStatus(error.message, 'error');
        }
    }

    renderReport(report) {
        const summary = report.summary;
        const hasIssues = report.catastrophic_warnings.length > 0 || report.misalignments.length > 0;

        // Determine overall status color
        let statusClass = 'status-good';
        let statusText = 'Good';
        if (summary.overall_score < 50) {
            statusClass = 'status-critical';
            statusText = 'Critical Issues';
        } else if (summary.overall_score < 80) {
            statusClass = 'status-warning';
            statusText = 'Needs Review';
        }

        let html = `
            <div class="audit-summary">
                <div class="audit-score ${statusClass}">
                    <div class="score-value">${summary.overall_score.toFixed(0)}</div>
                    <div class="score-label">Alignment Score</div>
                    <div class="score-status">${statusText}</div>
                </div>
                <div class="audit-stats">
                    <div class="stat">
                        <span class="stat-value">${summary.total_claims}</span>
                        <span class="stat-label">Claims Extracted</span>
                    </div>
                    <div class="stat stat-good">
                        <span class="stat-value">${summary.aligned_count}</span>
                        <span class="stat-label">Verified</span>
                    </div>
                    <div class="stat stat-bad">
                        <span class="stat-value">${summary.misaligned_count}</span>
                        <span class="stat-label">Misaligned</span>
                    </div>
                    <div class="stat stat-neutral">
                        <span class="stat-value">${summary.unverified_count}</span>
                        <span class="stat-label">Unverified</span>
                    </div>
                </div>
                <div class="audit-recommendation">
                    <strong>Recommendation:</strong> ${this.escapeHtml(summary.recommendation)}
                </div>
            </div>
        `;

        // Catastrophic warnings section
        if (report.catastrophic_warnings.length > 0) {
            html += `
                <div class="audit-section-header">
                    <h4>Critical Warnings (${report.catastrophic_warnings.length})</h4>
                </div>
                <div class="warning-list">
            `;

            // Group by tier
            const tier1Warnings = report.catastrophic_warnings.filter(w => w.tier === 'tier1' && !w.suppressed);
            const tier2Warnings = report.catastrophic_warnings.filter(w => w.tier === 'tier2' && !w.suppressed);
            const otherWarnings = report.catastrophic_warnings.filter(w => !['tier1', 'tier2'].includes(w.tier) && !w.suppressed);
            const suppressedWarnings = report.catastrophic_warnings.filter(w => w.suppressed);

            if (tier1Warnings.length > 0) {
                html += `<div class="tier-group tier1-group">
                    <div class="tier-header">Tier 1: Results Invalid (${tier1Warnings.length})</div>
                    ${tier1Warnings.map(w => this.renderWarning(w)).join('')}
                </div>`;
            }

            if (tier2Warnings.length > 0) {
                html += `<div class="tier-group tier2-group">
                    <div class="tier-header">Tier 2: Results Questionable (${tier2Warnings.length})</div>
                    ${tier2Warnings.map(w => this.renderWarning(w)).join('')}
                </div>`;
            }

            if (otherWarnings.length > 0) {
                html += `<div class="tier-group other-group">
                    <div class="tier-header">Other Issues (${otherWarnings.length})</div>
                    ${otherWarnings.map(w => this.renderWarning(w)).join('')}
                </div>`;
            }

            if (suppressedWarnings.length > 0) {
                html += `<div class="tier-group suppressed-group collapsed">
                    <div class="tier-header toggle-header" onclick="this.parentElement.classList.toggle('collapsed')">
                        Suppressed (${suppressedWarnings.length}) <span class="toggle-icon">+</span>
                    </div>
                    <div class="tier-content">
                        ${suppressedWarnings.map(w => this.renderWarning(w, true)).join('')}
                    </div>
                </div>`;
            }

            html += '</div>';
        }

        // Misalignments section
        if (report.misalignments.length > 0) {
            html += `
                <div class="audit-section-header">
                    <h4>Misalignments (${report.misalignments.length})</h4>
                </div>
                <div class="misalignment-list">
                    ${report.misalignments.map(m => this.renderMisalignment(m)).join('')}
                </div>
            `;
        }

        // Verified claims section (collapsible)
        if (report.verified_claims.length > 0) {
            html += `
                <div class="audit-section-header collapsible collapsed" onclick="this.classList.toggle('collapsed'); this.nextElementSibling.classList.toggle('hidden')">
                    <h4>Verified Claims (${report.verified_claims.length}) <span class="toggle-icon">+</span></h4>
                </div>
                <div class="verified-list hidden">
                    ${report.verified_claims.map(v => this.renderVerifiedClaim(v)).join('')}
                </div>
            `;
        }

        // Unverified claims section (collapsible)
        if (report.unverified_claims.length > 0) {
            html += `
                <div class="audit-section-header collapsible collapsed" onclick="this.classList.toggle('collapsed'); this.nextElementSibling.classList.toggle('hidden')">
                    <h4>Unverified Claims (${report.unverified_claims.length}) <span class="toggle-icon">+</span></h4>
                </div>
                <div class="unverified-list hidden">
                    ${report.unverified_claims.map(u => this.renderUnverifiedClaim(u)).join('')}
                </div>
            `;
        }

        this.elements.auditModalBody.innerHTML = html;

        // Bind false positive buttons
        this.elements.auditModalBody.querySelectorAll('.fp-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const warningId = btn.dataset.warningId;
                const warning = report.catastrophic_warnings.find(w => w.id === warningId);
                if (warning) {
                    this.openFpModal(warning);
                }
            });
        });
    }

    renderWarning(warning, isSuppressed = false) {
        const tierClass = `tier-${warning.tier}`;
        const categoryLabel = this.formatCategory(warning.category);

        return `
            <div class="warning-item ${tierClass} ${isSuppressed ? 'suppressed' : ''}">
                <div class="warning-header">
                    <span class="warning-category">${categoryLabel}</span>
                    <span class="warning-pattern">${this.escapeHtml(warning.pattern_type)}</span>
                    ${!isSuppressed ? `<button class="fp-btn" data-warning-id="${warning.id}" title="Mark as false positive">FP</button>` : ''}
                </div>
                <div class="warning-location">
                    <span class="file-path">${this.escapeHtml(warning.relative_path)}</span>
                    <span class="line-numbers">:${warning.line_start}-${warning.line_end}</span>
                </div>
                <div class="warning-description">${this.escapeHtml(warning.description)}</div>
                <div class="warning-why">
                    <strong>Why:</strong> ${this.escapeHtml(warning.why_catastrophic)}
                </div>
                <div class="warning-recommendation">
                    <strong>Fix:</strong> ${this.escapeHtml(warning.recommendation)}
                </div>
                <div class="warning-code">
                    <pre><code>${this.escapeHtml(warning.code_snippet)}</code></pre>
                </div>
                ${isSuppressed ? `<div class="suppression-reason">Suppressed: ${this.escapeHtml(warning.suppression_reason || 'No reason provided')}</div>` : ''}
            </div>
        `;
    }

    renderMisalignment(misalignment) {
        const tierClass = `tier-${misalignment.tier}`;
        const claim = misalignment.claim;

        return `
            <div class="misalignment-item ${tierClass}">
                <div class="misalignment-claim">
                    <strong>Claim:</strong> "${this.escapeHtml(claim.text)}"
                    ${claim.expected_value ? `<span class="expected-value">(expected: ${this.escapeHtml(claim.expected_value)})</span>` : ''}
                </div>
                <div class="misalignment-explanation">
                    ${this.escapeHtml(misalignment.explanation)}
                </div>
                ${misalignment.specific_issues.length > 0 ? `
                    <div class="misalignment-issues">
                        <strong>Issues:</strong>
                        <ul>
                            ${misalignment.specific_issues.map(i => `<li>${this.escapeHtml(i)}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
                ${misalignment.matched_code.length > 0 ? `
                    <div class="matched-code">
                        <strong>Related code:</strong>
                        ${misalignment.matched_code.map(c => `
                            <div class="code-location">${this.escapeHtml(c.relative_path)}:${c.start_line}-${c.end_line}</div>
                        `).join('')}
                    </div>
                ` : ''}
            </div>
        `;
    }

    renderVerifiedClaim(result) {
        return `
            <div class="verified-item">
                <div class="verified-claim">${this.escapeHtml(result.claim.text)}</div>
                <div class="verified-explanation">${this.escapeHtml(result.explanation)}</div>
            </div>
        `;
    }

    renderUnverifiedClaim(result) {
        return `
            <div class="unverified-item">
                <div class="unverified-claim">${this.escapeHtml(result.claim.text)}</div>
                <div class="unverified-reason">${this.escapeHtml(result.explanation)}</div>
            </div>
        `;
    }

    formatCategory(category) {
        const labels = {
            'data_leakage': 'Data Leakage',
            'evaluation_error': 'Evaluation Error',
            'training_issue': 'Training Issue',
            'reproducibility': 'Reproducibility',
            'data_integrity': 'Data Integrity',
            'implementation_bug': 'Implementation Bug',
            'statistical_error': 'Statistical Error',
        };
        return labels[category] || category;
    }

    // False Positive Management
    openFpModal(warning) {
        this.pendingFpWarningId = warning.id;
        this.elements.fpWarningDescription.textContent = `${warning.pattern_type}: ${warning.description}`;
        this.elements.fpReason.value = '';
        this.elements.fpScope.value = 'this_audit';
        this.elements.fpModal.classList.remove('hidden');
    }

    closeFpModal() {
        this.pendingFpWarningId = null;
        this.elements.fpModal.classList.add('hidden');
    }

    async submitFalsePositive() {
        if (!this.pendingFpWarningId || !this.currentAuditId) return;

        const reason = this.elements.fpReason.value.trim();
        if (!reason) {
            alert('Please provide a reason');
            return;
        }

        const scope = this.elements.fpScope.value;

        try {
            await api.markFalsePositive(
                this.currentAuditId,
                this.pendingFpWarningId,
                reason,
                scope
            );

            // Update the warning in the current report
            const warning = this.currentReport.catastrophic_warnings.find(
                w => w.id === this.pendingFpWarningId
            );
            if (warning) {
                warning.suppressed = true;
                warning.suppression_reason = reason;
            }

            this.closeFpModal();
            this.renderReport(this.currentReport);
            this.setStatus('Marked as false positive', 'success');
        } catch (error) {
            alert(error.message);
        }
    }

    // History
    async toggleHistoryMenu() {
        const isHidden = this.elements.auditHistoryMenu.classList.contains('hidden');

        if (isHidden) {
            await this.loadHistory();
            this.elements.auditHistoryMenu.classList.remove('hidden');
        } else {
            this.elements.auditHistoryMenu.classList.add('hidden');
        }
    }

    async loadHistory() {
        try {
            const data = await api.getAuditHistory();
            this.renderHistory(data.audits);
        } catch (error) {
            this.elements.auditHistoryMenu.innerHTML = '<div class="history-empty">Failed to load history</div>';
        }
    }

    renderHistory(audits) {
        if (!audits || audits.length === 0) {
            this.elements.auditHistoryMenu.innerHTML = '<div class="history-empty">No previous audits</div>';
            return;
        }

        this.elements.auditHistoryMenu.innerHTML = audits.slice(0, 10).map(audit => {
            const scoreClass = audit.score >= 80 ? 'good' : audit.score >= 50 ? 'warning' : 'bad';
            const scoreDisplay = audit.score !== null ? `${audit.score.toFixed(0)}%` : '--';

            return `
                <div class="history-item" data-id="${audit.audit_id}">
                    <div class="history-item-info">
                        <div class="history-item-name">${this.escapeHtml(audit.codebase_name || 'Unknown')}</div>
                        <div class="history-item-meta">
                            <span class="audit-score-badge ${scoreClass}">${scoreDisplay}</span>
                            ${this.formatDate(audit.timestamp)}
                        </div>
                    </div>
                    <button class="history-item-delete" data-id="${audit.audit_id}" title="Delete">x</button>
                </div>
            `;
        }).join('');

        // Bind click events
        this.elements.auditHistoryMenu.querySelectorAll('.history-item').forEach(item => {
            item.addEventListener('click', async (e) => {
                if (e.target.classList.contains('history-item-delete')) return;
                const auditId = item.dataset.id;
                await this.showAuditReport(auditId);
                this.elements.auditHistoryMenu.classList.add('hidden');
            });
        });

        this.elements.auditHistoryMenu.querySelectorAll('.history-item-delete').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const auditId = btn.dataset.id;
                await this.deleteAudit(auditId);
            });
        });
    }

    async deleteAudit(auditId) {
        try {
            await api.deleteAudit(auditId);
            await this.loadHistory();
        } catch (error) {
            console.error('Failed to delete audit:', error);
        }
    }

    // Utilities
    openModal() {
        this.elements.auditModal.classList.remove('hidden');
    }

    closeModal() {
        this.elements.auditModal.classList.add('hidden');
    }

    setStatus(message, type = '') {
        this.elements.auditStatus.textContent = message;
        this.elements.auditStatus.className = `status ${type}`;
    }

    formatDate(isoString) {
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

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.auditManager = new AuditManager();
});
