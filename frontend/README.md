# Nexus Trading Frontend

Production-ready React + TypeScript frontend for the Nexus Trading System with broker-safe architecture.

## 🚀 Features

- **React 18 + TypeScript** - Modern development stack
- **Tailwind CSS** - Utility-first styling with custom design system
- **React Query** - Server state management and caching
- **Zustand** - Lightweight client state management
- **React Router v6** - Client-side routing
- **React Hook Form + Zod** - Form validation and management
- **JWT Authentication** - Secure token-based auth with refresh mechanism
- **Real-time Updates** - WebSocket integration for live data
- **Mobile Responsive** - Tablet and mobile optimized design
- **Production Ready** - Docker deployment with Nginx

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   └── ui/           # Base UI components (Button, Input, etc.)
│   ├── pages/              # Page components
│   │   ├── Login.tsx
│   │   ├── Dashboard.tsx
│   │   ├── TradeForm.tsx
│   │   └── Reconciliation.tsx
│   ├── hooks/              # Custom React hooks
│   │   ├── useAuth.ts
│   │   ├── useTrading.ts
│   │   └── useReconciliation.ts
│   ├── services/           # API service layers
│   │   ├── api.ts
│   │   ├── auth.ts
│   │   ├── trading.ts
│   │   └── reconciliation.ts
│   ├── store/              # State management
│   │   ├── authStore.ts
│   │   └── uiStore.ts
│   ├── types/              # TypeScript type definitions
│   │   └── api.ts
│   ├── utils/              # Utility functions
│   │   ├── cn.ts
│   │   └── formatters.ts
│   ├── App.tsx             # Main application component
│   ├── main.tsx            # Application entry point
│   └── index.css           # Global styles
├── public/                 # Static assets
├── Dockerfile             # Production container
├── nginx.conf             # Nginx configuration
├── package.json           # Dependencies and scripts
├── tsconfig.json         # TypeScript configuration
├── vite.config.ts        # Vite build configuration
├── tailwind.config.js    # Tailwind CSS configuration
└── .env.example          # Environment variables template
```

## 🛠️ Development Setup

### Prerequisites

- Node.js 18+
- npm 9+

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Quantumlorld/Nexus-trading-intelligence-network.git
cd nexus-trading-intelligence-network/frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Environment configuration**
```bash
cp .env.example .env
# Edit .env with your API configuration
```

4. **Start development server**
```bash
npm run dev
```

The application will be available at `http://localhost:3000`

## 🏗️ Build & Deployment

### Development Build

```bash
npm run build
```

### Production Build with Docker

1. **Build and run with Docker Compose**
```bash
docker-compose -f docker-compose-broker-safe.yml up -d
```

2. **Build standalone Docker image**
```bash
docker build -t nexus-frontend .
docker run -p 80:80 nexus-frontend
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | Backend API base URL | `http://localhost:8002` |
| `VITE_API_VERSION` | API version | `v1` |
| `VITE_WS_URL` | WebSocket URL | `ws://localhost:8002/ws` |
| `VITE_APP_NAME` | Application name | `Nexus Trading System` |
| `VITE_ENABLE_REAL_TIME_UPDATES` | Enable WebSocket updates | `true` |
| `VITE_DEFAULT_LEVERAGE` | Default trading leverage | `1:100` |

### API Integration

The frontend integrates with the following backend endpoints:

- **Authentication**: `/api/v1/auth/*`
- **Trading**: `/api/v1/trading/*`
- **Reconciliation**: `/api/v1/reconciliation/*`

## 🎨 Design System

### Colors

- **Primary**: Blue (#3b82f6)
- **Success**: Green (#22c55e)
- **Warning**: Yellow (#f59e0b)
- **Danger**: Red (#ef4444)

### Components

All components follow the atomic design pattern:
- **Base components** in `src/components/ui/`
- **Composite components** in `src/components/`
- **Consistent styling** with Tailwind classes
- **TypeScript props** with proper typing

## 🔐 Security Features

- **JWT Authentication** with automatic refresh
- **Session storage** (not localStorage for security)
- **Role-based access control**
- **CSRF protection** via HTTP headers
- **XSS protection** with content security policy
- **Secure headers** in Nginx configuration

## 📱 Responsive Design

- **Mobile-first** approach
- **Breakpoints**: sm (640px), md (768px), lg (1024px)
- **Touch-friendly** interactions
- **Optimized** performance for all devices

## 🧪 Testing

```bash
# Run linting
npm run lint

# Fix linting issues
npm run lint:fix

# Type checking
npm run type-check

# Format code
npm run format
```

## 📊 Performance

- **Code splitting** with Vite
- **Tree shaking** for unused code elimination
- **Lazy loading** for route components
- **Image optimization** with WebP support
- **Gzip compression** in production

## 🔍 Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 📝 Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |
| `npm run lint:fix` | Fix ESLint issues |
| `npm run format` | Format with Prettier |
| `npm run type-check` | Type checking |

## 🚀 Production Deployment

### Docker Production

1. **Build image**
```bash
docker build -t nexus-frontend:latest .
```

2. **Run container**
```bash
docker run -d \
  --name nexus-frontend \
  -p 80:80 \
  --env NODE_ENV=production \
  nexus-frontend:latest
```

### Environment Configuration

- **Development**: Vite dev server with HMR
- **Staging**: Built assets with staging API
- **Production**: Nginx serving optimized assets

## 🔧 Customization

### Adding New Pages

1. Create component in `src/pages/`
2. Add route in `src/App.tsx`
3. Update navigation if needed

### API Integration

1. Define types in `src/types/api.ts`
2. Create service in `src/services/`
3. Build custom hooks in `src/hooks/`

### Styling

1. Use Tailwind classes
2. Extend theme in `tailwind.config.js`
3. Create reusable components in `src/components/ui/`

## 🐛 Troubleshooting

### Common Issues

1. **Build fails**: Check Node.js version (18+ required)
2. **API errors**: Verify backend is running and CORS is configured
3. **Auth issues**: Check token storage and refresh mechanism
4. **Styling issues**: Clear browser cache and restart dev server

### Debug Mode

Enable debug mode in `.env`:
```bash
VITE_ENABLE_DEBUG=true
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the troubleshooting section
