# Uprez Valuation Platform - Frontend

A comprehensive Next.js 15 frontend application for AI-powered IPO valuation analysis.

## 🚀 Features

- **Modern Tech Stack**: Built with Next.js 15, TypeScript, and Tailwind CSS
- **AI-Powered Analysis**: Advanced valuation algorithms and scenario modeling
- **Real-time Collaboration**: WebSocket integration for live updates
- **Interactive Charts**: D3.js and Recharts for financial visualizations
- **Document Processing**: Drag-and-drop file uploads with progress tracking
- **State Management**: Zustand for global state, TanStack Query for server state
- **Form Validation**: React Hook Form with Zod schemas
- **Accessibility**: WCAG 2.1 compliant components
- **Multi-tenant**: White-labeling and branding support
- **Mobile Responsive**: Optimized for all device sizes

## 📁 Project Structure

```
src/
├── app/                    # Next.js 15 App Router
│   ├── auth/              # Authentication pages
│   ├── dashboard/         # Main dashboard
│   ├── projects/          # Project management
│   ├── valuations/        # Valuation workflows
│   └── reports/           # Report generation
├── components/            # Reusable components
│   ├── ui/               # Base UI components
│   ├── valuation/        # Valuation-specific components
│   ├── charts/           # Chart components
│   ├── forms/            # Form components
│   └── layout/           # Layout components
├── lib/                  # Utility libraries
│   ├── api.ts           # API client
│   ├── utils.ts         # Helper functions
│   ├── websocket.ts     # WebSocket client
│   └── validations.ts   # Zod schemas
├── stores/               # Zustand stores
│   ├── auth.ts          # Authentication state
│   ├── projects.ts      # Projects state
│   ├── theme.ts         # Theme/branding state
│   └── collaboration.ts # Real-time collaboration
└── types/               # TypeScript type definitions
```

## 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd uprez-valuation/src/frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your configuration
   ```

4. **Start development server**
   ```bash
   npm run dev
   ```

## 🚀 Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking
- `npm test` - Run Jest tests
- `npm run test:watch` - Run tests in watch mode
- `npm run test:coverage` - Generate test coverage report

## 🏗 Key Components

### ValuationMetricCard
Interactive metric cards with real-time updates, trend indicators, and confidence levels.

```tsx
<ValuationMetricCard
  title="Portfolio Value"
  value={2400000000}
  previousValue={2100000000}
  format="currency"
  showTrend
  confidenceLevel={0.85}
/>
```

### InteractiveValuationChart
Advanced financial charts with multiple visualization types and scenario comparison.

```tsx
<InteractiveValuationChart
  data={chartData}
  title="Valuation Trends"
  type="area"
  scenarios={scenarios}
  showBrush
/>
```

### ScenarioModeler
Comprehensive scenario modeling interface with sliders, assumptions management, and results visualization.

```tsx
<ScenarioModeler
  scenarios={scenarios}
  onScenariosChange={setScenarios}
  onRunValuation={runValuation}
  baseMetrics={baseMetrics}
/>
```

### DocumentUploader
Drag-and-drop file upload with progress tracking, file validation, and error handling.

```tsx
<DocumentUploader
  onFilesUpload={handleFilesUpload}
  maxFiles={10}
  maxSize={50 * 1024 * 1024} // 50MB
  acceptedTypes={['application/pdf', '...']}
/>
```

### WaterfallChart
Specialized waterfall charts for valuation bridge analysis.

```tsx
<WaterfallChart
  data={waterfallData}
  title="Valuation Bridge"
  showConnectors
/>
```

## 🎨 Styling & Theming

The application uses Tailwind CSS with a custom design system:

- **Color Palette**: Customizable primary/secondary colors
- **Dark Mode**: Full dark mode support
- **Multi-tenant Branding**: Organization-specific themes
- **Responsive Design**: Mobile-first approach
- **Animations**: Framer Motion for smooth transitions

## 🔐 Authentication & Security

- JWT-based authentication with refresh tokens
- Role-based access control (Admin, Analyst, Viewer)
- Secure API communication with automatic token management
- Session persistence with automatic cleanup

## 📊 State Management

### Zustand Stores
- **Auth Store**: User authentication, roles, and sessions
- **Projects Store**: Project management and CRUD operations
- **Theme Store**: Branding, colors, and UI preferences
- **Collaboration Store**: Real-time user presence and activities

### TanStack Query
- Server state management with caching
- Optimistic updates and error handling
- Background refetching and synchronization

## 🔌 Real-time Features

WebSocket integration for:
- Live collaboration and user presence
- Real-time document processing updates
- Calculation progress and completion notifications
- Team activity feeds

## 📱 Accessibility

- WCAG 2.1 AA compliance
- Keyboard navigation support
- Screen reader compatibility
- High contrast mode support
- Focus management and ARIA labels

## 🧪 Testing

The application includes comprehensive testing:

- **Unit Tests**: Component testing with Jest and React Testing Library
- **Integration Tests**: API integration and user flow testing
- **E2E Tests**: Cypress for end-to-end testing
- **Visual Regression**: Chromatic for visual testing

## 📈 Performance

- **Code Splitting**: Automatic route-based code splitting
- **Image Optimization**: Next.js Image component with WebP support
- **Bundle Analysis**: Webpack bundle analyzer integration
- **Caching**: Aggressive caching strategies for static assets
- **Lazy Loading**: Component and route lazy loading

## 🔧 Development Guidelines

### Code Standards
- TypeScript strict mode enabled
- ESLint and Prettier configuration
- Conventional commit messages
- Component documentation with JSDoc

### Component Architecture
- Composition over inheritance
- Custom hooks for shared logic
- Proper separation of concerns
- Consistent prop interfaces

### Performance Best Practices
- Memoization for expensive calculations
- Virtualization for large lists
- Debounced search and input handlers
- Optimized re-renders with React.memo

## 🚀 Deployment

### Production Build
```bash
npm run build
npm run start
```

### Docker
```bash
docker build -t uprez-frontend .
docker run -p 3000:3000 uprez-frontend
```

### Environment Variables
Ensure all required environment variables are set for production deployment.

## 📚 API Integration

The frontend integrates with a comprehensive REST API:
- Automatic error handling and retry logic
- Request/response interceptors for authentication
- Type-safe API client with TypeScript
- File upload with progress tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is proprietary software. All rights reserved.

## 🆘 Support

For support and questions:
- Email: support@uprez.com
- Documentation: https://docs.uprez.com
- Slack: #uprez-support