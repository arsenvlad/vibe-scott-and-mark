// Scott and Mark Podcast Analytics - JavaScript

class PodcastAnalytics {
    constructor() {
        this.apiBase = '';
        this.episodes = [];
        this.statistics = {};
        this.charts = {};
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadData();
    }
    
    setupEventListeners() {
        // Fetch new episodes button
        document.getElementById('fetchEpisodesBtn').addEventListener('click', () => {
            this.fetchNewEpisodes();
        });
        
        // Process all audio button
        document.getElementById('processAllBtn').addEventListener('click', () => {
            this.processAllAudio();
        });
        
        // Modal close
        document.querySelector('.modal-close').addEventListener('click', () => {
            this.closeModal();
        });
        
        // Close modal on backdrop click
        document.getElementById('episodeModal').addEventListener('click', (e) => {
            if (e.target === e.currentTarget) {
                this.closeModal();
            }
        });
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });
    }
    
    async loadData() {
        this.showLoading('Loading podcast data...');
        
        try {
            // Load episodes and statistics
            await Promise.all([
                this.loadEpisodes(),
                this.loadStatistics()
            ]);
            
            this.updateUI();
            this.createCharts();
        } catch (error) {
            console.error('Error loading data:', error);
            this.showError('Failed to load podcast data');
        } finally {
            this.hideLoading();
        }
    }
    
    async loadEpisodes() {
        try {
            const response = await fetch('/api/episodes');
            const data = await response.json();
            
            if (data.success) {
                this.episodes = data.episodes;
            } else {
                throw new Error(data.error || 'Failed to load episodes');
            }
        } catch (error) {
            console.error('Error loading episodes:', error);
            throw error;
        }
    }
    
    async loadStatistics() {
        try {
            const response = await fetch('/api/statistics');
            const data = await response.json();
            
            if (data.success) {
                this.statistics = data.statistics;
            } else {
                throw new Error(data.error || 'Failed to load statistics');
            }
        } catch (error) {
            console.error('Error loading statistics:', error);
            throw error;
        }
    }
    
    async fetchNewEpisodes() {
        const button = document.getElementById('fetchEpisodesBtn');
        const originalText = button.innerHTML;
        
        try {
            button.disabled = true;
            button.innerHTML = '<i class=\"fas fa-spinner fa-spin\"></i> Fetching...';
            
            this.showLoading('Fetching new episodes from YouTube...');
            
            const response = await fetch('/api/fetch-new-episodes', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showSuccess(`Found ${data.new_episodes} new episodes!`);
                await this.loadData(); // Reload data to show new episodes
            } else {
                throw new Error(data.error || 'Failed to fetch episodes');
            }
        } catch (error) {
            console.error('Error fetching episodes:', error);
            this.showError('Failed to fetch new episodes');
        } finally {
            button.disabled = false;
            button.innerHTML = originalText;
            this.hideLoading();
        }
    }
    
    async processEpisodeAudio(videoId) {
        try {
            this.showLoading(`Processing audio for episode...`);
            
            // Get the episode details
            const episode = this.episodes.find(ep => ep.video_id === videoId);
            if (!episode) {
                throw new Error('Episode not found');
            }
            
            // Use the new real audio processing endpoint
            const response = await fetch('/api/process-episode', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    video_url: episode.video_url,
                    video_id: videoId
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showSuccess('Audio processing completed!');
                await this.loadData(); // Reload to show updated analysis
                return data.results;
            } else {
                throw new Error(data.error || 'Failed to process audio');
            }
        } catch (error) {
            console.error('Error processing audio:', error);
            this.showError('Failed to process episode audio: ' + error.message);
            throw error;
        } finally {
            this.hideLoading();
        }
    }
    
    async processAllAudio() {
        const unprocessedEpisodes = this.episodes.filter(ep => 
            !ep.total_words && !ep.scott_words && !ep.mark_words
        );
        
        if (unprocessedEpisodes.length === 0) {
            this.showInfo('All episodes have already been processed!');
            return;
        }
        
        const button = document.getElementById('processAllBtn');
        const originalText = button.innerHTML;
        
        try {
            button.disabled = true;
            button.innerHTML = '<i class=\"fas fa-spinner fa-spin\"></i> Processing...';
            
            for (let i = 0; i < unprocessedEpisodes.length; i++) {
                const episode = unprocessedEpisodes[i];
                this.showLoading(`Processing episode ${i + 1} of ${unprocessedEpisodes.length}: ${episode.title}`);
                
                try {
                    await this.processEpisodeAudio(episode.video_id);
                } catch (error) {
                    console.error(`Failed to process episode ${episode.video_id}:`, error);
                    continue; // Continue with next episode
                }
            }
            
            this.showSuccess(`Processed ${unprocessedEpisodes.length} episodes!`);
            await this.loadData(); // Reload all data
            
        } catch (error) {
            console.error('Error processing all audio:', error);
            this.showError('Failed to process all episodes');
        } finally {
            button.disabled = false;
            button.innerHTML = originalText;
            this.hideLoading();
        }
    }
    
    updateUI() {
        this.updateStatistics();
        this.renderEpisodes();
    }
    
    updateStatistics() {
        // Update stat cards
        document.getElementById('totalEpisodes').textContent = this.statistics.total_episodes || 0;
        document.getElementById('totalViews').textContent = this.formatNumber(this.statistics.total_views || 0);
        document.getElementById('totalHours').textContent = this.statistics.total_duration_hours || 0;
        document.getElementById('totalWords').textContent = this.formatNumber(this.statistics.total_words || 0);
    }
    
    renderEpisodes() {
        const container = document.getElementById('episodesList');
        
        if (!this.episodes.length) {
            container.innerHTML = `
                <div class=\"text-center\" style=\"grid-column: 1 / -1; padding: 2rem;\">
                    <i class=\"fas fa-podcast\" style=\"font-size: 3rem; color: var(--text-muted); margin-bottom: 1rem;\"></i>
                    <p>No episodes found. Click \"Fetch New Episodes\" to get started!</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = this.episodes.map(episode => this.createEpisodeCard(episode)).join('');
        
        // Add event listeners for episode actions
        this.setupEpisodeEventListeners();
    }
    
    createEpisodeCard(episode) {
        const isProcessed = episode.total_words > 0;
        const processingStatus = isProcessed ? 'processed' : 'pending';
        const statusText = isProcessed ? 'Analyzed' : 'Not Analyzed';
        
        return `
            <div class=\"episode-card fade-in\" data-video-id=\"${episode.video_id}\">
                <img src=\"${episode.thumbnail}\" alt=\"${episode.title}\" class=\"episode-thumbnail\">
                <div class=\"episode-content\">
                    <h3 class=\"episode-title\">${episode.title}</h3>
                    
                    <div class=\"episode-meta\">
                        <div class=\"episode-meta-item\">
                            <i class=\"fas fa-calendar\"></i>
                            <span>${episode.upload_date_formatted}</span>
                        </div>
                        <div class=\"episode-meta-item\">
                            <i class=\"fas fa-clock\"></i>
                            <span>${episode.duration_formatted}</span>
                        </div>
                        <div class=\"episode-meta-item\">
                            <i class=\"fas fa-eye\"></i>
                            <span>${this.formatNumber(episode.view_count)} views</span>
                        </div>
                    </div>
                    
                    <div class=\"processing-status status-${processingStatus}\">
                        <i class=\"fas fa-${isProcessed ? 'check-circle' : 'clock'}\"></i>
                        ${statusText}
                    </div>
                    
                    ${isProcessed ? `
                        <div class=\"episode-stats\">
                            <div class=\"episode-stat\">
                                <div class=\"episode-stat-value\">${this.formatNumber(episode.total_words || 0)}</div>
                                <div class=\"episode-stat-label\">Total Words</div>
                            </div>
                            <div class=\"episode-stat\">
                                <div class=\"episode-stat-value\">${episode.scott_percentage || 0}%</div>
                                <div class=\"episode-stat-label\">Scott</div>
                            </div>
                        </div>
                    ` : ''}
                    
                    <div class=\"episode-actions\">
                        <button class=\"btn btn-secondary episode-details-btn\" data-video-id=\"${episode.video_id}\">
                            <i class=\"fas fa-info-circle\"></i>
                            Details
                        </button>
                        ${!isProcessed ? `
                            <button class=\"btn btn-primary episode-process-btn\" data-video-id=\"${episode.video_id}\">
                                <i class=\"fas fa-cogs\"></i>
                                Analyze
                            </button>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }
    
    setupEpisodeEventListeners() {
        // Episode details buttons
        document.querySelectorAll('.episode-details-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const videoId = btn.getAttribute('data-video-id');
                this.showEpisodeDetails(videoId);
            });
        });
        
        // Episode process buttons
        document.querySelectorAll('.episode-process-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const videoId = btn.getAttribute('data-video-id');
                await this.processEpisodeAudio(videoId);
            });
        });
        
        // Episode cards (click to show details)
        document.querySelectorAll('.episode-card').forEach(card => {
            card.addEventListener('click', () => {
                const videoId = card.getAttribute('data-video-id');
                this.showEpisodeDetails(videoId);
            });
        });
    }
    
    showEpisodeDetails(videoId) {
        const episode = this.episodes.find(ep => ep.video_id === videoId);
        if (!episode) return;
        
        const modal = document.getElementById('episodeModal');
        const modalTitle = document.getElementById('modalTitle');
        const modalContent = document.getElementById('modalContent');
        
        modalTitle.textContent = episode.title;
        
        const isProcessed = episode.total_words > 0;
        
        modalContent.innerHTML = `
            <div class=\"episode-details\">
                <div class=\"episode-meta\">
                    <div class=\"episode-meta-item\">
                        <i class=\"fas fa-calendar\"></i>
                        <span>Uploaded: ${episode.upload_date_formatted}</span>
                    </div>
                    <div class=\"episode-meta-item\">
                        <i class=\"fas fa-clock\"></i>
                        <span>Duration: ${episode.duration_formatted}</span>
                    </div>
                    <div class=\"episode-meta-item\">
                        <i class=\"fas fa-eye\"></i>
                        <span>Views: ${this.formatNumber(episode.view_count)}</span>
                    </div>
                    <div class=\"episode-meta-item\">
                        <i class=\"fas fa-user\"></i>
                        <span>Channel: ${episode.uploader}</span>
                    </div>
                </div>
                
                ${episode.description ? `
                    <div class=\"episode-description\">
                        <h4>Description</h4>
                        <p>${episode.description.substring(0, 300)}${episode.description.length > 300 ? '...' : ''}</p>
                    </div>
                ` : ''}
                
                ${isProcessed ? `
                    <div class=\"analysis-results\">
                        <h4>Audio Analysis Results</h4>
                        <div class=\"stats-grid\">
                            <div class=\"stat-card\">
                                <div class=\"stat-content\">
                                    <h3>${this.formatNumber(episode.total_words)}</h3>
                                    <p>Total Words</p>
                                </div>
                            </div>
                            <div class=\"stat-card\">
                                <div class=\"stat-content\">
                                    <h3>${this.formatNumber(episode.scott_words)}</h3>
                                    <p>Scott's Words (${episode.scott_percentage}%)</p>
                                </div>
                            </div>
                            <div class=\"stat-card\">
                                <div class=\"stat-content\">
                                    <h3>${this.formatNumber(episode.mark_words)}</h3>
                                    <p>Mark's Words (${episode.mark_percentage}%)</p>
                                </div>
                            </div>
                        </div>
                    </div>
                ` : `
                    <div class=\"analysis-placeholder\">
                        <p>This episode hasn't been analyzed yet. Click \"Analyze\" to process the audio and get speaking statistics.</p>
                        <button class=\"btn btn-primary\" onclick=\"app.processEpisodeAudio('${episode.video_id}')\">
                            <i class=\"fas fa-cogs\"></i>
                            Analyze Episode
                        </button>
                    </div>
                `}
                
                <div class=\"episode-links\">
                    <a href=\"${episode.url}\" target=\"_blank\" class=\"btn btn-primary\">
                        <i class=\"fab fa-youtube\"></i>
                        Watch on YouTube
                    </a>
                </div>
            </div>
        `;
        
        modal.style.display = 'block';
        document.body.style.overflow = 'hidden';
    }
    
    closeModal() {
        const modal = document.getElementById('episodeModal');
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
    }
    
    createCharts() {
        // Destroy existing charts first
        this.destroyExistingCharts();
        
        this.createViewsChart();
        this.createSpeakingChart();
        this.createDurationChart();
        this.createWordsChart();
    }
    
    destroyExistingCharts() {
        // Destroy existing Chart.js instances to prevent canvas reuse errors
        if (this.charts.views) {
            this.charts.views.destroy();
            this.charts.views = null;
        }
        if (this.charts.speaking) {
            this.charts.speaking.destroy();
            this.charts.speaking = null;
        }
        if (this.charts.duration) {
            this.charts.duration.destroy();
            this.charts.duration = null;
        }
        if (this.charts.words) {
            this.charts.words.destroy();
            this.charts.words = null;
        }
    }
    
    createViewsChart() {
        const ctx = document.getElementById('viewsChart');
        if (!ctx) return;
        
        // Destroy existing chart if it exists
        if (this.charts.views) {
            this.charts.views.destroy();
        }
        
        // Prepare data for views over time (newest episodes on the right)
        const sortedEpisodes = [...this.episodes].sort((a, b) => {
            // Convert YYYYMMDD to YYYY-MM-DD for proper date parsing
            const dateA = a.upload_date.substr(0,4) + '-' + a.upload_date.substr(4,2) + '-' + a.upload_date.substr(6,2);
            const dateB = b.upload_date.substr(0,4) + '-' + b.upload_date.substr(4,2) + '-' + b.upload_date.substr(6,2);
            return new Date(dateA) - new Date(dateB);
        });
        
        const data = {
            labels: sortedEpisodes.map(ep => ep.upload_date_formatted),
            datasets: [{
                label: 'Views',
                data: sortedEpisodes.map(ep => ep.view_count),
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.4,
                fill: true
            }]
        };
        
        this.charts.views = new Chart(ctx, {
            type: 'line',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: (value) => this.formatNumber(value)
                        }
                    },
                    x: {
                        ticks: {
                            maxTicksLimit: 8
                        }
                    }
                }
            }
        });
    }
    
    createSpeakingChart() {
        const ctx = document.getElementById('speakingChart');
        if (!ctx) return;
        
        // Destroy existing chart if it exists
        if (this.charts.speaking) {
            this.charts.speaking.destroy();
        }
        
        const scottTotal = this.statistics.scott_words || 0;
        const markTotal = this.statistics.mark_words || 0;
        
        if (scottTotal === 0 && markTotal === 0) {
            const context = ctx.getContext('2d');
            context.clearRect(0, 0, ctx.width, ctx.height);
            context.font = '14px Inter, sans-serif';
            context.fillStyle = '#64748b';
            context.textAlign = 'center';
            context.fillText('No analyzed episodes yet', ctx.width / 2, ctx.height / 2);
            return;
        }
        
        const data = {
            labels: ['Scott', 'Mark'],
            datasets: [{
                data: [scottTotal, markTotal],
                backgroundColor: ['#3b82f6', '#ef4444'],
                borderWidth: 0
            }]
        };
        
        this.charts.speaking = new Chart(ctx, {
            type: 'doughnut',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    createDurationChart() {
        const ctx = document.getElementById('durationChart');
        if (!ctx) return;
        
        // Destroy existing chart if it exists
        if (this.charts.duration) {
            this.charts.duration.destroy();
        }
        
        const sortedEpisodes = [...this.episodes].sort((a, b) => {
            // Convert YYYYMMDD to YYYY-MM-DD for proper date parsing
            const dateA = a.upload_date.substr(0,4) + '-' + a.upload_date.substr(4,2) + '-' + a.upload_date.substr(6,2);
            const dateB = b.upload_date.substr(0,4) + '-' + b.upload_date.substr(4,2) + '-' + b.upload_date.substr(6,2);
            return new Date(dateA) - new Date(dateB);
        });
        
        const data = {
            labels: sortedEpisodes.map(ep => ep.title.substring(0, 20) + '...'),
            datasets: [{
                label: 'Duration (minutes)',
                data: sortedEpisodes.map(ep => Math.round(ep.duration_seconds / 60)),
                backgroundColor: '#10b981',
                borderColor: '#059669',
                borderWidth: 1
            }]
        };
        
        this.charts.duration = new Chart(ctx, {
            type: 'bar',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Minutes'
                        }
                    },
                    x: {
                        ticks: {
                            maxTicksLimit: 10
                        }
                    }
                }
            }
        });
    }
    
    createWordsChart() {
        const ctx = document.getElementById('wordsChart');
        if (!ctx) return;
        
        // Destroy existing chart if it exists
        if (this.charts.words) {
            this.charts.words.destroy();
        }
        
        const processedEpisodes = this.episodes.filter(ep => ep.total_words > 0);
        
        if (processedEpisodes.length === 0) {
            const context = ctx.getContext('2d');
            context.clearRect(0, 0, ctx.width, ctx.height);
            context.font = '14px Inter, sans-serif';
            context.fillStyle = '#64748b';
            context.textAlign = 'center';
            context.fillText('No analyzed episodes yet', ctx.width / 2, ctx.height / 2);
            return;
        }
        
        const data = {
            labels: processedEpisodes.map(ep => ep.title.substring(0, 20) + '...'),
            datasets: [
                {
                    label: 'Scott',
                    data: processedEpisodes.map(ep => ep.scott_words),
                    backgroundColor: '#3b82f6'
                },
                {
                    label: 'Mark',
                    data: processedEpisodes.map(ep => ep.mark_words),
                    backgroundColor: '#ef4444'
                }
            ]
        };
        
        this.charts.words = new Chart(ctx, {
            type: 'bar',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        stacked: true,
                        ticks: {
                            maxTicksLimit: 8
                        }
                    },
                    y: {
                        stacked: true,
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Words'
                        }
                    }
                }
            }
        });
    }
    
    // Utility methods
    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }
    
    showLoading(message = 'Loading...') {
        const loader = document.getElementById('loadingIndicator');
        const text = document.getElementById('loadingText');
        text.textContent = message;
        loader.style.display = 'block';
    }
    
    hideLoading() {
        const loader = document.getElementById('loadingIndicator');
        loader.style.display = 'none';
    }
    
    showError(message) {
        this.showNotification(message, 'error');
    }
    
    showSuccess(message) {
        this.showNotification(message, 'success');
    }
    
    showInfo(message) {
        this.showNotification(message, 'info');
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class=\"notification-content\">
                <i class=\"fas fa-${type === 'error' ? 'exclamation-triangle' : type === 'success' ? 'check-circle' : 'info-circle'}\"></i>
                <span>${message}</span>
            </div>
            <button class=\"notification-close\">&times;</button>
        `;
        
        // Add styles if not already added
        if (!document.querySelector('#notification-styles')) {
            const styles = document.createElement('style');
            styles.id = 'notification-styles';
            styles.textContent = `
                .notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    z-index: 10000;
                    padding: 1rem 1.5rem;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    color: white;
                    max-width: 400px;
                    animation: slideIn 0.3s ease;
                }
                .notification-error { background-color: #ef4444; }
                .notification-success { background-color: #10b981; }
                .notification-info { background-color: #3b82f6; }
                .notification-content {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }
                .notification-close {
                    background: none;
                    border: none;
                    color: white;
                    font-size: 1.2rem;
                    cursor: pointer;
                    margin-left: 1rem;
                }
                @keyframes slideIn {
                    from { transform: translateX(100%); }
                    to { transform: translateX(0); }
                }
            `;
            document.head.appendChild(styles);
        }
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
        
        // Close button
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.remove();
        });
    }
}

// Initialize the application
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new PodcastAnalytics();
});
