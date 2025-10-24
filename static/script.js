/**
 * AI Chatbot Web Interface JavaScript
 * Handles chat interactions, API calls, and UI updates
 */

class ChatbotInterface {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.clearButton = document.getElementById('clearButton');
        this.settingsButton = document.getElementById('settingsButton');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.loadingIndicator = document.getElementById('loadingIndicator');
        
        // Settings modal elements
        this.settingsModal = document.getElementById('settingsModal');
        this.closeSettings = document.getElementById('closeSettings');
        this.saveSettings = document.getElementById('saveSettings');
        this.thresholdSlider = document.getElementById('thresholdSlider');
        this.thresholdValue = document.getElementById('thresholdValue');
        this.sentimentToggle = document.getElementById('sentimentToggle');
        this.debugToggle = document.getElementById('debugToggle');
        
        this.isLoading = false;
        this.debugMode = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.checkStatus();
        this.loadSettings();
    }
    
    setupEventListeners() {
        // Send message on button click or Enter key
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Clear chat
        this.clearButton.addEventListener('click', () => this.clearChat());
        
        // Settings modal
        this.settingsButton.addEventListener('click', () => this.openSettings());
        this.closeSettings.addEventListener('click', () => this.closeSettingsModal());
        this.saveSettings.addEventListener('click', () => this.saveSettings());
        
        // Threshold slider
        this.thresholdSlider.addEventListener('input', (e) => {
            this.thresholdValue.textContent = e.target.value;
        });
        
        // Debug toggle
        this.debugToggle.addEventListener('change', (e) => {
            this.debugMode = e.target.checked;
        });
        
        // Close modal on outside click
        this.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.settingsModal) {
                this.closeSettingsModal();
            }
        });
        
        // Auto-resize input
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = this.messageInput.scrollHeight + 'px';
        });
    }
    
    async checkStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.status === 'ready') {
                this.updateStatus('connected', 'Connected');
            } else {
                this.updateStatus('disconnected', 'Not Ready');
            }
        } catch (error) {
            this.updateStatus('disconnected', 'Connection Error');
            console.error('Status check failed:', error);
        }
    }
    
    updateStatus(status, text) {
        const statusDot = this.statusIndicator.querySelector('.status-dot');
        const statusText = this.statusIndicator.querySelector('.status-text');
        
        statusDot.className = `status-dot ${status}`;
        statusText.textContent = text;
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isLoading) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        
        // Show loading indicator
        this.showLoading();
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.addMessage(data.response, 'bot', data);
            } else {
                this.addMessage(data.response || 'Sorry, I encountered an error.', 'bot');
            }
        } catch (error) {
            this.addMessage('Sorry, I encountered a connection error. Please try again.', 'bot');
            console.error('Chat error:', error);
        } finally {
            this.hideLoading();
        }
    }
    
    addMessage(text, sender, data = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        messageText.textContent = text;
        
        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = new Date().toLocaleTimeString();
        
        content.appendChild(messageText);
        content.appendChild(messageTime);
        
        // Add debug info if available and debug mode is on
        if (data && this.debugMode && sender === 'bot') {
            const debugInfo = this.createDebugInfo(data);
            content.appendChild(debugInfo);
        }
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    createDebugInfo(data) {
        const debugDiv = document.createElement('div');
        debugDiv.className = 'debug-info';
        
        let debugText = '';
        if (data.intent) {
            debugText += `Intent: ${data.intent}`;
        }
        if (data.confidence) {
            debugText += ` | Confidence: ${(data.confidence * 100).toFixed(1)}%`;
        }
        if (data.sentiment && data.sentiment.combined) {
            debugText += ` | Sentiment: ${data.sentiment.combined.sentiment}`;
        }
        
        debugDiv.textContent = debugText;
        return debugDiv;
    }
    
    showLoading() {
        this.isLoading = true;
        this.sendButton.disabled = true;
        this.loadingIndicator.classList.add('show');
    }
    
    hideLoading() {
        this.isLoading = false;
        this.sendButton.disabled = false;
        this.loadingIndicator.classList.remove('show');
    }
    
    clearChat() {
        this.chatMessages.innerHTML = `
            <div class="message bot-message">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="message-text">
                        Hello! I'm your AI assistant. How can I help you today? 🤖
                    </div>
                    <div class="message-time">Just now</div>
                </div>
            </div>
        `;
        
        // Call API to clear chat history
        fetch('/api/clear', { method: 'POST' });
    }
    
    openSettings() {
        this.settingsModal.classList.add('show');
        this.settingsModal.style.display = 'flex';
    }
    
    closeSettingsModal() {
        this.settingsModal.classList.remove('show');
        setTimeout(() => {
            this.settingsModal.style.display = 'none';
        }, 300);
    }
    
    async saveSettings() {
        const threshold = parseFloat(this.thresholdSlider.value);
        const sentiment = this.sentimentToggle.checked;
        
        try {
            // Update threshold
            await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ threshold: threshold })
            });
            
            // Update sentiment
            await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sentiment: sentiment })
            });
            
            // Save to localStorage
            localStorage.setItem('chatbotSettings', JSON.stringify({
                threshold: threshold,
                sentiment: sentiment,
                debug: this.debugToggle.checked
            }));
            
            this.closeSettingsModal();
            this.showNotification('Settings saved successfully!');
        } catch (error) {
            console.error('Failed to save settings:', error);
            this.showNotification('Failed to save settings', 'error');
        }
    }
    
    loadSettings() {
        const saved = localStorage.getItem('chatbotSettings');
        if (saved) {
            const settings = JSON.parse(saved);
            this.thresholdSlider.value = settings.threshold || 0.75;
            this.thresholdValue.textContent = settings.threshold || 0.75;
            this.sentimentToggle.checked = settings.sentiment !== false;
            this.debugToggle.checked = settings.debug || false;
            this.debugMode = settings.debug || false;
        }
    }
    
    showNotification(message, type = 'success') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'error' ? '#f87171' : '#4ade80'};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            z-index: 1001;
            animation: slideInRight 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100%);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideOutRight {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100%);
        }
    }
`;
document.head.appendChild(style);

// Initialize the chatbot interface when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatbotInterface();
});

