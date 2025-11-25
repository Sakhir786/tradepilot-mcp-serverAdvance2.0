"""
TradePilot Alerts & Notification System
========================================
Multi-channel notification system for trade alerts.

Supported Channels:
- Discord (webhook)
- Slack (webhook)
- Telegram (bot API)
- Custom webhooks
- Email (SMTP)
- Console logging

Features:
- Alert templates for different signal types
- Rate limiting to prevent spam
- Alert history tracking
- Custom alert rules
- Position invalidation alerts

Author: TradePilot Integration
"""

import os
import json
import requests
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import time
import hashlib


class AlertPriority(Enum):
    """Alert priority levels"""
    CRITICAL = "CRITICAL"    # Immediate action required
    HIGH = "HIGH"            # Important alert
    NORMAL = "NORMAL"        # Standard alert
    LOW = "LOW"              # Informational


class AlertType(Enum):
    """Types of alerts"""
    NEW_SETUP = "NEW_SETUP"
    ENTRY_TRIGGER = "ENTRY_TRIGGER"
    TARGET_HIT = "TARGET_HIT"
    STOP_HIT = "STOP_HIT"
    INVALIDATION = "INVALIDATION"
    SCAN_COMPLETE = "SCAN_COMPLETE"
    SYSTEM = "SYSTEM"
    CUSTOM = "CUSTOM"


class NotificationChannel(Enum):
    """Notification channels"""
    DISCORD = "DISCORD"
    SLACK = "SLACK"
    TELEGRAM = "TELEGRAM"
    WEBHOOK = "WEBHOOK"
    EMAIL = "EMAIL"
    CONSOLE = "CONSOLE"


@dataclass
class Alert:
    """Individual alert"""
    id: str
    type: AlertType
    priority: AlertPriority
    title: str
    message: str
    timestamp: str
    
    # Trade details (if applicable)
    ticker: Optional[str] = None
    direction: Optional[str] = None
    action: Optional[str] = None
    confidence: Optional[str] = None
    win_probability: Optional[float] = None
    playbook: Optional[str] = None
    
    # Option details
    strike: Optional[float] = None
    delta: Optional[float] = None
    expiry_dte: Optional[int] = None
    
    # Execution details
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    current_price: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    channels_sent: List[str] = field(default_factory=list)
    sent_successfully: bool = False
    error: Optional[str] = None


@dataclass
class ChannelConfig:
    """Configuration for a notification channel"""
    channel_type: NotificationChannel
    enabled: bool = True
    webhook_url: Optional[str] = None
    api_key: Optional[str] = None
    chat_id: Optional[str] = None  # For Telegram
    email_config: Optional[Dict] = None
    rate_limit_seconds: float = 1.0
    priority_filter: List[AlertPriority] = field(default_factory=lambda: list(AlertPriority))


class AlertTemplates:
    """Pre-built alert templates"""
    
    @staticmethod
    def new_setup_discord(alert: Alert) -> Dict:
        """Discord embed for new setup alert"""
        color = 0x00FF00 if alert.direction == "BULLISH" else 0xFF0000 if alert.direction == "BEARISH" else 0x808080
        emoji = "🟢" if alert.direction == "BULLISH" else "🔴" if alert.direction == "BEARISH" else "⚪"
        
        return {
            "embeds": [{
                "title": f"{emoji} {alert.action} Alert: {alert.ticker}",
                "description": alert.message,
                "color": color,
                "fields": [
                    {"name": "Direction", "value": alert.direction or "N/A", "inline": True},
                    {"name": "Confidence", "value": f"{alert.confidence} ({alert.win_probability:.0f}%)" if alert.win_probability else "N/A", "inline": True},
                    {"name": "Playbook", "value": alert.playbook or "N/A", "inline": True},
                    {"name": "Strike", "value": f"${alert.strike:.2f}" if alert.strike else "N/A", "inline": True},
                    {"name": "Delta", "value": f"{alert.delta:.2f}" if alert.delta else "N/A", "inline": True},
                    {"name": "Expiry", "value": f"{alert.expiry_dte} DTE" if alert.expiry_dte else "N/A", "inline": True},
                    {"name": "Entry", "value": f"${alert.entry_price:.2f}" if alert.entry_price else "N/A", "inline": True},
                    {"name": "Target", "value": f"${alert.target_price:.2f}" if alert.target_price else "N/A", "inline": True},
                    {"name": "Stop", "value": f"${alert.stop_price:.2f}" if alert.stop_price else "N/A", "inline": True},
                ],
                "footer": {"text": f"TradePilot | {alert.timestamp}"},
                "timestamp": datetime.now().isoformat()
            }]
        }
    
    @staticmethod
    def invalidation_discord(alert: Alert) -> Dict:
        """Discord embed for invalidation alert"""
        return {
            "embeds": [{
                "title": f"🚨 INVALIDATION: {alert.ticker}",
                "description": alert.message,
                "color": 0xFFA500,
                "fields": [
                    {"name": "Current Price", "value": f"${alert.current_price:.2f}" if alert.current_price else "N/A", "inline": True},
                    {"name": "Original Action", "value": alert.action or "N/A", "inline": True},
                    {"name": "Reason", "value": alert.metadata.get("reason", "Price moved against position")},
                ],
                "footer": {"text": f"TradePilot Alert | {alert.timestamp}"},
                "timestamp": datetime.now().isoformat()
            }]
        }
    
    @staticmethod
    def scan_complete_discord(alert: Alert) -> Dict:
        """Discord embed for scan completion"""
        return {
            "embeds": [{
                "title": "🔍 Scan Complete",
                "description": alert.message,
                "color": 0x3498DB,
                "fields": [
                    {"name": "Setups Found", "value": str(alert.metadata.get("setups_found", 0)), "inline": True},
                    {"name": "Bullish", "value": str(alert.metadata.get("bullish_count", 0)), "inline": True},
                    {"name": "Bearish", "value": str(alert.metadata.get("bearish_count", 0)), "inline": True},
                ],
                "footer": {"text": f"TradePilot Scanner | {alert.timestamp}"},
                "timestamp": datetime.now().isoformat()
            }]
        }
    
    @staticmethod
    def new_setup_slack(alert: Alert) -> Dict:
        """Slack block for new setup alert"""
        emoji = ":chart_with_upwards_trend:" if alert.direction == "BULLISH" else ":chart_with_downwards_trend:"
        
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"{emoji} {alert.action}: {alert.ticker}"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": alert.message}
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Direction:* {alert.direction}"},
                        {"type": "mrkdwn", "text": f"*Confidence:* {alert.confidence} ({alert.win_probability:.0f}%)"},
                        {"type": "mrkdwn", "text": f"*Playbook:* {alert.playbook or 'N/A'}"},
                        {"type": "mrkdwn", "text": f"*Strike:* ${alert.strike:.2f}" if alert.strike else "*Strike:* N/A"},
                    ]
                },
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": f"TradePilot | {alert.timestamp}"}]
                }
            ]
        }
    
    @staticmethod
    def new_setup_telegram(alert: Alert) -> str:
        """Telegram message for new setup alert"""
        emoji = "🟢" if alert.direction == "BULLISH" else "🔴"
        
        return f"""
{emoji} *{alert.action}: {alert.ticker}*

*Direction:* {alert.direction}
*Confidence:* {alert.confidence} ({alert.win_probability:.0f}%)
*Playbook:* {alert.playbook or 'N/A'}

📊 *Option Details:*
• Strike: ${alert.strike:.2f if alert.strike else 'N/A'}
• Delta: {alert.delta:.2f if alert.delta else 'N/A'}
• Expiry: {alert.expiry_dte} DTE

💰 *Execution:*
• Entry: ${alert.entry_price:.2f if alert.entry_price else 'N/A'}
• Target: ${alert.target_price:.2f if alert.target_price else 'N/A'}
• Stop: ${alert.stop_price:.2f if alert.stop_price else 'N/A'}

_{alert.timestamp}_
"""


class TradePilotAlerts:
    """
    Main alert system for TradePilot
    
    Usage:
        alerts = TradePilotAlerts()
        alerts.add_channel(NotificationChannel.DISCORD, webhook_url="...")
        alerts.send_new_setup_alert(analysis_result)
    """
    
    def __init__(self):
        """Initialize the alert system"""
        self._channels: Dict[NotificationChannel, ChannelConfig] = {}
        self._alert_history: List[Alert] = []
        self._rate_limits: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        # Alert queue for async processing
        self._alert_queue = queue.Queue()
        self._worker_thread = None
        self._running = False
        
        # Duplicate detection
        self._recent_alerts: Dict[str, datetime] = {}
        self._duplicate_window_seconds = 300  # 5 minutes
        
        # Load from environment
        self._load_from_env()
    
    def _load_from_env(self):
        """Load channel configurations from environment variables"""
        # Discord
        discord_url = os.environ.get('TRADEPILOT_DISCORD_WEBHOOK')
        if discord_url:
            self.add_channel(NotificationChannel.DISCORD, webhook_url=discord_url)
        
        # Slack
        slack_url = os.environ.get('TRADEPILOT_SLACK_WEBHOOK')
        if slack_url:
            self.add_channel(NotificationChannel.SLACK, webhook_url=slack_url)
        
        # Telegram
        telegram_token = os.environ.get('TRADEPILOT_TELEGRAM_TOKEN')
        telegram_chat = os.environ.get('TRADEPILOT_TELEGRAM_CHAT_ID')
        if telegram_token and telegram_chat:
            self.add_channel(
                NotificationChannel.TELEGRAM, 
                api_key=telegram_token,
                chat_id=telegram_chat
            )
    
    def add_channel(self, 
                    channel_type: NotificationChannel,
                    webhook_url: Optional[str] = None,
                    api_key: Optional[str] = None,
                    chat_id: Optional[str] = None,
                    enabled: bool = True,
                    rate_limit_seconds: float = 1.0,
                    priority_filter: Optional[List[AlertPriority]] = None) -> 'TradePilotAlerts':
        """
        Add a notification channel
        
        Args:
            channel_type: Type of channel (DISCORD, SLACK, etc.)
            webhook_url: Webhook URL for Discord/Slack
            api_key: API key/token for Telegram
            chat_id: Chat ID for Telegram
            enabled: Whether channel is enabled
            rate_limit_seconds: Minimum seconds between alerts
            priority_filter: Only send alerts matching these priorities
            
        Returns:
            self for chaining
        """
        config = ChannelConfig(
            channel_type=channel_type,
            enabled=enabled,
            webhook_url=webhook_url,
            api_key=api_key,
            chat_id=chat_id,
            rate_limit_seconds=rate_limit_seconds,
            priority_filter=priority_filter or list(AlertPriority)
        )
        
        self._channels[channel_type] = config
        return self
    
    def remove_channel(self, channel_type: NotificationChannel) -> 'TradePilotAlerts':
        """Remove a notification channel"""
        if channel_type in self._channels:
            del self._channels[channel_type]
        return self
    
    def start_async_worker(self):
        """Start background worker for async alert processing"""
        if not self._running:
            self._running = True
            self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
            self._worker_thread.start()
    
    def stop_async_worker(self):
        """Stop background worker"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
    
    def _process_queue(self):
        """Background worker to process alert queue"""
        while self._running:
            try:
                alert = self._alert_queue.get(timeout=1)
                self._send_alert_to_channels(alert)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Alerts] Queue processing error: {e}")
    
    def _generate_alert_id(self, alert_type: AlertType, ticker: Optional[str], action: Optional[str]) -> str:
        """Generate unique alert ID"""
        components = [alert_type.value, ticker or "", action or "", datetime.now().isoformat()]
        hash_input = "|".join(components)
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is a duplicate"""
        # Create fingerprint
        fingerprint = f"{alert.type.value}|{alert.ticker}|{alert.direction}|{alert.playbook}"
        
        with self._lock:
            if fingerprint in self._recent_alerts:
                last_sent = self._recent_alerts[fingerprint]
                if (datetime.now() - last_sent).total_seconds() < self._duplicate_window_seconds:
                    return True
            
            self._recent_alerts[fingerprint] = datetime.now()
            
            # Clean old entries
            cutoff = datetime.now() - timedelta(seconds=self._duplicate_window_seconds)
            self._recent_alerts = {k: v for k, v in self._recent_alerts.items() if v > cutoff}
        
        return False
    
    def _can_send_to_channel(self, channel: NotificationChannel, priority: AlertPriority) -> bool:
        """Check if we can send to channel (rate limiting + priority filter)"""
        config = self._channels.get(channel)
        if not config or not config.enabled:
            return False
        
        if priority not in config.priority_filter:
            return False
        
        # Rate limiting
        rate_key = f"{channel.value}"
        with self._lock:
            last_sent = self._rate_limits.get(rate_key, 0)
            if time.time() - last_sent < config.rate_limit_seconds:
                return False
            self._rate_limits[rate_key] = time.time()
        
        return True
    
    def _send_discord(self, alert: Alert, config: ChannelConfig) -> bool:
        """Send alert to Discord"""
        try:
            if not config.webhook_url:
                return False
            
            # Select template based on alert type
            if alert.type == AlertType.NEW_SETUP:
                payload = AlertTemplates.new_setup_discord(alert)
            elif alert.type == AlertType.INVALIDATION:
                payload = AlertTemplates.invalidation_discord(alert)
            elif alert.type == AlertType.SCAN_COMPLETE:
                payload = AlertTemplates.scan_complete_discord(alert)
            else:
                payload = {"content": f"**{alert.title}**\n{alert.message}"}
            
            response = requests.post(
                config.webhook_url,
                json=payload,
                timeout=10
            )
            
            return response.status_code in [200, 204]
            
        except Exception as e:
            print(f"[Alerts] Discord send error: {e}")
            return False
    
    def _send_slack(self, alert: Alert, config: ChannelConfig) -> bool:
        """Send alert to Slack"""
        try:
            if not config.webhook_url:
                return False
            
            if alert.type == AlertType.NEW_SETUP:
                payload = AlertTemplates.new_setup_slack(alert)
            else:
                payload = {"text": f"*{alert.title}*\n{alert.message}"}
            
            response = requests.post(
                config.webhook_url,
                json=payload,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"[Alerts] Slack send error: {e}")
            return False
    
    def _send_telegram(self, alert: Alert, config: ChannelConfig) -> bool:
        """Send alert to Telegram"""
        try:
            if not config.api_key or not config.chat_id:
                return False
            
            if alert.type == AlertType.NEW_SETUP:
                text = AlertTemplates.new_setup_telegram(alert)
            else:
                text = f"*{alert.title}*\n\n{alert.message}"
            
            url = f"https://api.telegram.org/bot{config.api_key}/sendMessage"
            payload = {
                "chat_id": config.chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            print(f"[Alerts] Telegram send error: {e}")
            return False
    
    def _send_webhook(self, alert: Alert, config: ChannelConfig) -> bool:
        """Send alert to custom webhook"""
        try:
            if not config.webhook_url:
                return False
            
            payload = {
                "alert_id": alert.id,
                "type": alert.type.value,
                "priority": alert.priority.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "ticker": alert.ticker,
                "direction": alert.direction,
                "action": alert.action,
                "confidence": alert.confidence,
                "win_probability": alert.win_probability,
                "playbook": alert.playbook,
                "option": {
                    "strike": alert.strike,
                    "delta": alert.delta,
                    "expiry_dte": alert.expiry_dte
                },
                "execution": {
                    "entry": alert.entry_price,
                    "target": alert.target_price,
                    "stop": alert.stop_price,
                    "current": alert.current_price
                },
                "metadata": alert.metadata
            }
            
            headers = {}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"
            
            response = requests.post(
                config.webhook_url,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            return response.status_code in [200, 201, 202, 204]
            
        except Exception as e:
            print(f"[Alerts] Webhook send error: {e}")
            return False
    
    def _send_console(self, alert: Alert, config: ChannelConfig) -> bool:
        """Print alert to console"""
        emoji_map = {
            AlertPriority.CRITICAL: "🚨",
            AlertPriority.HIGH: "⚠️",
            AlertPriority.NORMAL: "📢",
            AlertPriority.LOW: "ℹ️"
        }
        
        emoji = emoji_map.get(alert.priority, "📢")
        print(f"\n{emoji} [{alert.priority.value}] {alert.title}")
        print(f"   {alert.message}")
        if alert.ticker:
            print(f"   Ticker: {alert.ticker} | {alert.direction} | {alert.action}")
        print(f"   Time: {alert.timestamp}\n")
        
        return True
    
    def _send_alert_to_channels(self, alert: Alert):
        """Send alert to all configured channels"""
        for channel_type, config in self._channels.items():
            if not self._can_send_to_channel(channel_type, alert.priority):
                continue
            
            success = False
            
            if channel_type == NotificationChannel.DISCORD:
                success = self._send_discord(alert, config)
            elif channel_type == NotificationChannel.SLACK:
                success = self._send_slack(alert, config)
            elif channel_type == NotificationChannel.TELEGRAM:
                success = self._send_telegram(alert, config)
            elif channel_type == NotificationChannel.WEBHOOK:
                success = self._send_webhook(alert, config)
            elif channel_type == NotificationChannel.CONSOLE:
                success = self._send_console(alert, config)
            
            if success:
                alert.channels_sent.append(channel_type.value)
        
        alert.sent_successfully = len(alert.channels_sent) > 0
        self._alert_history.append(alert)
    
    def send_alert(self, alert: Alert, async_send: bool = False):
        """
        Send an alert
        
        Args:
            alert: Alert to send
            async_send: Whether to send asynchronously
        """
        if self._is_duplicate(alert):
            return
        
        if async_send:
            self._alert_queue.put(alert)
        else:
            self._send_alert_to_channels(alert)
    
    def send_new_setup_alert(self, 
                            analysis_result: Any,
                            priority: AlertPriority = AlertPriority.NORMAL) -> Alert:
        """
        Send alert for a new trading setup
        
        Args:
            analysis_result: FullAnalysisResult from engine
            priority: Alert priority
            
        Returns:
            Created Alert
        """
        alert = Alert(
            id=self._generate_alert_id(AlertType.NEW_SETUP, analysis_result.ticker, analysis_result.action),
            type=AlertType.NEW_SETUP,
            priority=priority,
            title=f"New Setup: {analysis_result.ticker}",
            message=f"{analysis_result.action} signal detected with {analysis_result.confidence.value} confidence",
            timestamp=datetime.now().isoformat(),
            ticker=analysis_result.ticker,
            direction=analysis_result.direction,
            action=analysis_result.action,
            confidence=analysis_result.confidence.value,
            win_probability=analysis_result.win_probability,
            playbook=analysis_result.matched_playbook,
            strike=analysis_result.strike,
            delta=analysis_result.delta,
            expiry_dte=analysis_result.expiry_dte,
            entry_price=analysis_result.entry_price,
            target_price=analysis_result.target_price,
            stop_price=analysis_result.stop_price,
            current_price=analysis_result.current_price
        )
        
        self.send_alert(alert)
        return alert
    
    def send_invalidation_alert(self,
                               ticker: str,
                               original_action: str,
                               current_price: float,
                               reason: str,
                               priority: AlertPriority = AlertPriority.HIGH) -> Alert:
        """Send alert when a trade setup is invalidated"""
        alert = Alert(
            id=self._generate_alert_id(AlertType.INVALIDATION, ticker, original_action),
            type=AlertType.INVALIDATION,
            priority=priority,
            title=f"Setup Invalidated: {ticker}",
            message=reason,
            timestamp=datetime.now().isoformat(),
            ticker=ticker,
            action=original_action,
            current_price=current_price,
            metadata={"reason": reason}
        )
        
        self.send_alert(alert)
        return alert
    
    def send_scan_complete_alert(self,
                                 scan_summary: Any,
                                 priority: AlertPriority = AlertPriority.LOW) -> Alert:
        """Send alert when a scan completes"""
        alert = Alert(
            id=self._generate_alert_id(AlertType.SCAN_COMPLETE, None, None),
            type=AlertType.SCAN_COMPLETE,
            priority=priority,
            title="Scan Complete",
            message=f"Found {scan_summary.setups_found} setups ({scan_summary.bullish_setups} bullish, {scan_summary.bearish_setups} bearish)",
            timestamp=datetime.now().isoformat(),
            metadata={
                "setups_found": scan_summary.setups_found,
                "bullish_count": scan_summary.bullish_setups,
                "bearish_count": scan_summary.bearish_setups,
                "tickers_scanned": scan_summary.tickers_scanned
            }
        )
        
        self.send_alert(alert)
        return alert
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get recent alert history"""
        return self._alert_history[-limit:]
    
    def clear_history(self):
        """Clear alert history"""
        self._alert_history = []


# Convenience function
def create_alerts(discord_webhook: Optional[str] = None,
                 slack_webhook: Optional[str] = None) -> TradePilotAlerts:
    """Create a new alerts instance"""
    alerts = TradePilotAlerts()
    
    if discord_webhook:
        alerts.add_channel(NotificationChannel.DISCORD, webhook_url=discord_webhook)
    if slack_webhook:
        alerts.add_channel(NotificationChannel.SLACK, webhook_url=slack_webhook)
    
    # Always add console
    alerts.add_channel(NotificationChannel.CONSOLE)
    
    return alerts


if __name__ == "__main__":
    # Example usage
    alerts = TradePilotAlerts()
    alerts.add_channel(NotificationChannel.CONSOLE)
    
    # Test alert
    test_alert = Alert(
        id="test123",
        type=AlertType.NEW_SETUP,
        priority=AlertPriority.HIGH,
        title="Test Alert",
        message="This is a test alert",
        timestamp=datetime.now().isoformat(),
        ticker="SPY",
        direction="BULLISH",
        action="BUY CALL",
        confidence="STRONG",
        win_probability=85.0
    )
    
    alerts.send_alert(test_alert)
    print("Alert system initialized successfully!")
