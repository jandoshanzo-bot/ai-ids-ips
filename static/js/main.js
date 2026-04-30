/**
 * IDS - Intrusion Detection System
 * Main JavaScript File
 */

// Global configuration
const IDS_CONFIG = {
    apiBaseUrl: '',
    alertThreshold: 0.7,
    refreshInterval: 30000 // 30 seconds
};

// Utility functions
const Utils = {
    /**
     * Format number as percentage
     */
    formatPercent: (value, decimals = 1) => {
        return (value * 100).toFixed(decimals) + '%';
    },

    /**
     * Format timestamp
     */
    formatTimestamp: (timestamp) => {
        if (!timestamp) return '-';
        const date = new Date(timestamp);
        return date.toLocaleString('kk-KZ');
    },

    /**
     * Debounce function
     */
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Show notification
     */
    showNotification: (message, type = 'info') => {
        const alertClass = {
            'success': 'alert-success',
            'error': 'alert-danger',
            'warning': 'alert-warning',
            'info': 'alert-info'
        }[type] || 'alert-info';

        const notification = document.createElement('div');
        notification.className = `alert ${alertClass} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(notification);

        // Auto dismiss after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
    },

    /**
     * Copy to clipboard
     */
    copyToClipboard: async (text) => {
        try {
            await navigator.clipboard.writeText(text);
            Utils.showNotification('Көшірілді!', 'success');
        } catch (err) {
            Utils.showNotification('Көшіру қатесі', 'error');
        }
    }
};

// API functions
const API = {
    /**
     * Make prediction
     */
    predict: async (features, model = 'random_forest') => {
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features, model })
            });
            return await response.json();
        } catch (error) {
            console.error('Prediction error:', error);
            return { success: false, error: error.message };
        }
    },

    /**
     * Batch prediction
     */
    predictBatch: async (data, model = 'random_forest') => {
        try {
            const response = await fetch('/api/predict/batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ data, model })
            });
            return await response.json();
        } catch (error) {
            console.error('Batch prediction error:', error);
            return { success: false, error: error.message };
        }
    },

    /**
     * Get statistics
     */
    getStatistics: async () => {
        try {
            const response = await fetch('/api/statistics');
            return await response.json();
        } catch (error) {
            console.error('Statistics error:', error);
            return { success: false, error: error.message };
        }
    },

    /**
     * Get history
     */
    getHistory: async (limit = 100) => {
        try {
            const response = await fetch(`/api/history?limit=${limit}`);
            return await response.json();
        } catch (error) {
            console.error('History error:', error);
            return { success: false, error: error.message };
        }
    },

    /**
     * Test Telegram
     */
    testTelegram: async () => {
        try {
            const response = await fetch('/api/telegram/test', {
                method: 'POST'
            });
            return await response.json();
        } catch (error) {
            console.error('Telegram test error:', error);
            return { success: false, error: error.message };
        }
    },

    /**
     * Health check
     */
    healthCheck: async () => {
        try {
            const response = await fetch('/api/health');
            return await response.json();
        } catch (error) {
            console.error('Health check error:', error);
            return { status: 'unhealthy', error: error.message };
        }
    }
};

// UI Components
const UI = {
    /**
     * Update statistics cards
     */
    updateStats: (stats) => {
        const elements = {
            total: document.querySelector('[data-stat="total"]'),
            attacks: document.querySelector('[data-stat="attacks"]'),
            normal: document.querySelector('[data-stat="normal"]'),
            alerts: document.querySelector('[data-stat="alerts"]')
        };

        if (elements.total) elements.total.textContent = stats.total || 0;
        if (elements.attacks) elements.attacks.textContent = stats.attacks || 0;
        if (elements.normal) elements.normal.textContent = stats.normal || 0;
        if (elements.alerts) elements.alerts.textContent = stats.alerts_sent || 0;
    },

    /**
     * Create prediction badge
     */
    createPredictionBadge: (prediction) => {
        const isAttack = prediction === 'Attack';
        return `
            <span class="badge ${isAttack ? 'badge-attack' : 'badge-normal'}">
                <i class="fas ${isAttack ? 'fa-exclamation-triangle' : 'fa-check-circle'} me-1"></i>
                ${isAttack ? 'Шабуыл' : 'Қалыпты'}
            </span>
        `;
    },

    /**
     * Create probability bar
     */
    createProbabilityBar: (probability) => {
        let colorClass = 'bg-success';
        if (probability > 0.7) colorClass = 'bg-danger';
        else if (probability > 0.3) colorClass = 'bg-warning';

        return `
            <div class="progress" style="height: 20px; width: 100px;">
                <div class="progress-bar ${colorClass}" style="width: ${probability * 100}%">
                    ${Utils.formatPercent(probability)}
                </div>
            </div>
        `;
    },

    /**
     * Animate number counter
     */
    animateCounter: (element, target, duration = 1000) => {
        const start = 0;
        const increment = target / (duration / 16);
        let current = start;

        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            element.textContent = Math.floor(current);
        }, 16);
    }
};

// Auto-refresh functionality
class AutoRefresh {
    constructor(interval = 30000) {
        this.interval = interval;
        this.timer = null;
    }

    start(callback) {
        this.stop();
        this.timer = setInterval(callback, this.interval);
    }

    stop() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Initialize tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipTriggerList.forEach(el => new bootstrap.Tooltip(el));

    // Initialize popovers
    const popoverTriggerList = document.querySelectorAll('[data-bs-toggle="popover"]');
    popoverTriggerList.forEach(el => new bootstrap.Popover(el));

    // Auto-refresh statistics on dashboard
    if (document.querySelector('[data-stat="total"]')) {
        const refresh = new AutoRefresh(30000);
        refresh.start(async () => {
            const result = await API.getStatistics();
            if (result.success) {
                UI.updateStats(result.statistics);
            }
        });
    }

    console.log('IDS initialized successfully!');
});

// Export for global access
window.IDS = {
    config: IDS_CONFIG,
    utils: Utils,
    api: API,
    ui: UI,
    AutoRefresh
};
