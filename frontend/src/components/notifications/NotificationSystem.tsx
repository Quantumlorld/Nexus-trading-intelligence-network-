import React, { useEffect, useState } from 'react';
import { toast } from 'react-hot-toast';
import { useAuthStore } from '@/store/authStore';

interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  action_url?: string;
}

interface WebSocketMessage {
  type: 'notification' | 'trade_update' | 'risk_alert' | 'system_alert';
  data: any;
}

const NotificationSystem: React.FC = () => {
  const { isAuthenticated } = useAuthStore();
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [ws, setWs] = useState<WebSocket | null>(null);

  // WebSocket connection for real-time notifications
  useEffect(() => {
    if (!isAuthenticated) {
      if (ws) {
        ws.close();
        setWs(null);
      }
      return;
    }

    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8002/ws';
    const websocket = new WebSocket(`${wsUrl}/notifications`);

    websocket.onopen = () => {
      console.log('WebSocket connected for notifications');
    };

    websocket.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        handleWebSocketMessage(message);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      // Attempt to reconnect after 5 seconds
      setTimeout(() => {
        if (isAuthenticated) {
          setWs(websocket);
        }
      }, 5000);
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    setWs(websocket);

    return () => {
      websocket.close();
    };
  }, [isAuthenticated]);

  const handleWebSocketMessage = (message: WebSocketMessage) => {
    switch (message.type) {
      case 'notification':
        handleNotification(message.data);
        break;
      case 'trade_update':
        handleTradeUpdate(message.data);
        break;
      case 'risk_alert':
        handleRiskAlert(message.data);
        break;
      case 'system_alert':
        handleSystemAlert(message.data);
        break;
    }
  };

  const handleNotification = (data: any) => {
    const notification: Notification = {
      id: data.id || Date.now().toString(),
      type: data.type || 'info',
      title: data.title || 'Notification',
      message: data.message || '',
      timestamp: data.timestamp || new Date().toISOString(),
      read: false,
      action_url: data.action_url,
    };

    setNotifications(prev => [notification, ...prev]);

    // Show toast notification
    const toastOptions = {
      duration: data.type === 'error' ? 6000 : 4000,
      icon: getNotificationIcon(data.type),
    };

    if (data.action_url) {
      toast(
        (t) => (
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium">{data.title}</div>
              <div className="text-sm">{data.message}</div>
            </div>
            <button
              onClick={() => {
                window.location.href = data.action_url;
                toast.dismiss(t.id);
              }}
              className="ml-4 px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
            >
              View
            </button>
          </div>
        ),
        toastOptions
      );
    } else {
      toast(`${data.title}: ${data.message}`, toastOptions);
    }
  };

  const handleTradeUpdate = (data: any) => {
    const message = `Trade ${data.trade_id} ${data.status.toLowerCase()}`;
    toast.success(message, { duration: 3000 });
  };

  const handleRiskAlert = (data: any) => {
    const message = `Risk Alert: ${data.message}`;
    toast.error(message, { duration: 8000 });
  };

  const handleSystemAlert = (data: any) => {
    const message = `System: ${data.message}`;
    toast(data.severity === 'critical' ? 'error' : 'warning', message, {
      duration: data.severity === 'critical' ? 10000 : 6000,
    });
  };

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'success':
        return '✅';
      case 'warning':
        return '⚠️';
      case 'error':
        return '❌';
      case 'info':
      default:
        return 'ℹ️';
    }
  };

  const markAsRead = (notificationId: string) => {
    setNotifications(prev =>
      prev.map(n => (n.id === notificationId ? { ...n, read: true } : n))
    );
  };

  const markAllAsRead = () => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })));
  };

  const unreadCount = notifications.filter(n => !n.read).length;

  // This component doesn't render anything visible
  // It only handles the notification system logic
  return null;
};

export default NotificationSystem;
