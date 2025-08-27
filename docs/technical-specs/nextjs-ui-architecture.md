# Next.js UI Architecture for IPO Valuation SaaS Platform

## Executive Summary

This document outlines a comprehensive Next.js UI architecture for the Uprez IPO Valuation SaaS platform, incorporating latest 2024-2025 best practices for financial applications. The architecture emphasizes performance, security, accessibility, and enterprise-grade features while maintaining exceptional user experience.

## Table of Contents

1. [Technology Stack & Framework Selection](#technology-stack--framework-selection)
2. [Application Architecture](#application-architecture)
3. [Component Library & Design System](#component-library--design-system)
4. [Data Visualization Strategy](#data-visualization-strategy)
5. [State Management Architecture](#state-management-architecture)
6. [Form Handling & Validation](#form-handling--validation)
7. [Enterprise Requirements](#enterprise-requirements)
8. [Performance Optimization](#performance-optimization)
9. [Security & Compliance](#security--compliance)
10. [Implementation Roadmap](#implementation-roadmap)

---

## Technology Stack & Framework Selection

### Next.js 15 with App Router

**Rationale**: Next.js 15 App Router provides the ideal foundation for financial SaaS applications with its advanced features:

- **Server Components**: Reduce JavaScript bundle size by 76.7% for data-heavy financial interfaces
- **Streaming & Suspense**: Enable non-blocking data fetches for real-time financial metrics
- **Performance**: Up to 96.3% faster code updates and 45.8% faster initial route compile
- **SEO Optimization**: Essential for B2B SaaS visibility with server-side rendering
- **Type Safety**: Full TypeScript support for financial data accuracy

### Core Dependencies

```json
{
  "dependencies": {
    "next": "^15.2.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "typescript": "^5.6.0",
    "@tanstack/react-query": "^5.0.0",
    "zustand": "^4.5.0",
    "react-hook-form": "^7.50.0",
    "zod": "^3.22.0",
    "@hookform/resolvers": "^3.3.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "tailwindcss": "^3.4.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0",
    "eslint": "^8.0.0",
    "eslint-config-next": "^15.0.0"
  }
}
```

---

## Application Architecture

### App Router Structure

```
src/
├── app/                          # Next.js 15 App Router
│   ├── (auth)/                   # Route groups for authentication
│   │   ├── login/
│   │   └── register/
│   ├── (dashboard)/              # Main application routes
│   │   ├── dashboard/
│   │   ├── valuation/            # Core valuation workflow
│   │   └── reports/
│   ├── (tenant)/[tenant]/        # Multi-tenant routes
│   │   └── dashboard/
│   ├── api/                      # API routes
│   │   ├── auth/
│   │   ├── valuation/
│   │   └── uploads/
│   ├── globals.css
│   ├── layout.tsx                # Root layout
│   ├── loading.tsx               # Global loading UI
│   ├── error.tsx                 # Global error boundary
│   └── not-found.tsx             # 404 page
├── components/                   # Reusable components
│   ├── ui/                       # Base UI components
│   ├── charts/                   # Data visualization
│   ├── forms/                    # Form components
│   └── layouts/                  # Layout components
├── lib/                          # Utilities and configurations
│   ├── auth/                     # Authentication logic
│   ├── api/                      # API client functions
│   ├── validations/              # Zod schemas
│   └── utils/                    # Helper functions
├── hooks/                        # Custom React hooks
├── store/                        # State management
├── types/                        # TypeScript type definitions
└── styles/                       # Global styles and themes
```

### Server vs Client Component Strategy

**Server Components** (Default for financial data):
- Initial data fetching for valuation reports
- Static content rendering (layouts, headers)
- SEO-critical pages
- Financial metrics display
- Document processing results

**Client Components** (Interactive features):
- Real-time data updates
- Interactive charts and visualizations
- Form inputs and validation
- File upload interfaces
- User interactions (filtering, sorting)

### Streaming Strategy

```typescript
// Streaming implementation for valuation workflow
import { Suspense } from 'react'
import { ValuationMetrics } from '@/components/valuation/metrics'
import { PeerAnalysis } from '@/components/valuation/peer-analysis'
import { RiskAssessment } from '@/components/valuation/risk-assessment'

export default function ValuationDashboard() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <Suspense fallback={<MetricsSkeleton />}>
        <ValuationMetrics />
      </Suspense>
      
      <Suspense fallback={<PeerAnalysisSkeleton />}>
        <PeerAnalysis />
      </Suspense>
      
      <Suspense fallback={<RiskSkeleton />}>
        <RiskAssessment />
      </Suspense>
    </div>
  )
}
```

---

## Component Library & Design System

### Recommended Stack: Tailwind CSS + shadcn/ui

**Selection Rationale**:
- **Full Customization**: Component ownership for financial-specific needs
- **Type Safety**: TypeScript-first components with Zod validation
- **Performance**: Zero runtime CSS overhead
- **Accessibility**: Built-in ARIA support and keyboard navigation
- **Maintainability**: Direct code access for customization

### Component Architecture

```typescript
// Base component structure
interface BaseComponentProps {
  className?: string
  children?: React.ReactNode
  variant?: 'default' | 'outline' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
}

// Financial-specific components
interface MetricCardProps extends BaseComponentProps {
  title: string
  value: string | number
  change?: {
    value: number
    direction: 'up' | 'down' | 'neutral'
    period: string
  }
  confidence?: {
    level: number
    factors: string[]
  }
  format?: 'currency' | 'percentage' | 'number'
}
```

### Core UI Components

1. **ValuationMetricCard**: Display key financial metrics with confidence indicators
2. **InteractiveChart**: Recharts-based visualizations with real-time updates
3. **DocumentUploader**: Drag-and-drop file upload with progress tracking
4. **ScenarioModeler**: Interactive sliders for IPO scenario planning
5. **ValuationBridge**: Waterfall chart showing valuation methodology
6. **ComplianceGate**: Role-based access control component
7. **TenantSwitcher**: Multi-tenancy navigation component

### Design Tokens

```typescript
// Tailwind configuration for financial themes
export const theme = {
  colors: {
    primary: {
      50: '#f0f9ff',
      500: '#0ea5e9',
      900: '#0c4a6e'
    },
    success: {
      50: '#f0fdf4',
      500: '#22c55e',
      900: '#14532d'
    },
    warning: {
      50: '#fffbeb',
      500: '#f59e0b',
      900: '#92400e'
    },
    danger: {
      50: '#fef2f2',
      500: '#ef4444',
      900: '#7f1d1d'
    }
  },
  fontFamily: {
    sans: ['Inter', 'system-ui', 'sans-serif'],
    mono: ['JetBrains Mono', 'monospace']
  }
}
```

---

## Data Visualization Strategy

### Library Selection: Recharts + D3.js Hybrid Approach

**Primary**: Recharts for standard financial charts
- React-native integration
- TypeScript support
- Responsive design
- Minimal configuration

**Advanced**: D3.js for custom visualizations
- Complex financial models
- Interactive networks
- Custom animations
- Maximum flexibility

### Chart Components Architecture

```typescript
interface BaseChartProps {
  data: any[]
  width?: number
  height?: number
  className?: string
  loading?: boolean
  error?: string
}

// Specialized financial charts
interface ValuationBridgeProps extends BaseChartProps {
  baseMultiple: number
  adjustments: BridgeAdjustment[]
  targetMultiple: number
  onHover?: (adjustment: BridgeAdjustment) => void
}

interface PeerComparisonProps extends BaseChartProps {
  peers: CompanyMetrics[]
  subject: CompanyMetrics
  metrics: ('pe' | 'ev_ebitda' | 'price_sales')[]
}
```

### Real-Time Data Integration

```typescript
// Real-time chart updates with React Query
function useRealtimeFinancialData(symbol: string) {
  return useQuery({
    queryKey: ['financial-data', symbol],
    queryFn: () => fetchFinancialData(symbol),
    refetchInterval: 30000, // 30 seconds
    staleTime: 10000,
    retry: 3
  })
}

// Streaming updates implementation
function RealtimeChart({ symbol }: { symbol: string }) {
  const { data, isLoading, error } = useRealtimeFinancialData(symbol)
  
  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data}>
        <XAxis dataKey="timestamp" />
        <YAxis />
        <Tooltip />
        <Line 
          type="monotone" 
          dataKey="value" 
          stroke="#0ea5e9"
          strokeWidth={2}
          animationDuration={500}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
```

### Chart Performance Optimization

- **Virtualization**: For large datasets (>1000 points)
- **Memoization**: React.memo for expensive chart calculations
- **Canvas Rendering**: For real-time updates with high frequency
- **Data Sampling**: Intelligent data reduction for mobile devices

---

## State Management Architecture

### Hybrid Approach: Zustand + TanStack Query

**Server State**: TanStack Query for API data management
- Caching and synchronization
- Background updates
- Optimistic updates
- Error handling

**Client State**: Zustand for application state
- User preferences
- UI state
- Form state
- Navigation state

### State Architecture

```typescript
// Server state with React Query
const useValuationData = (companyId: string) => {
  return useQuery({
    queryKey: ['valuation', companyId],
    queryFn: () => api.valuation.get(companyId),
    staleTime: 5 * 60 * 1000, // 5 minutes
    cacheTime: 10 * 60 * 1000, // 10 minutes
    retry: 3,
    retryDelay: attemptIndex => Math.min(1000 * 2 ** attemptIndex, 30000)
  })
}

// Client state with Zustand
interface AppState {
  // User preferences
  theme: 'light' | 'dark'
  currency: 'AUD' | 'USD'
  
  // UI state
  sidebarOpen: boolean
  activeView: 'dashboard' | 'valuation' | 'reports'
  
  // Workflow state
  currentStep: number
  completedSteps: Set<number>
  
  // Actions
  setTheme: (theme: 'light' | 'dark') => void
  toggleSidebar: () => void
  setActiveView: (view: string) => void
  completeStep: (step: number) => void
}

const useAppStore = create<AppState>((set) => ({
  theme: 'light',
  currency: 'AUD',
  sidebarOpen: true,
  activeView: 'dashboard',
  currentStep: 1,
  completedSteps: new Set(),
  
  setTheme: (theme) => set({ theme }),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setActiveView: (activeView) => set({ activeView }),
  completeStep: (step) => set((state) => ({
    completedSteps: new Set(state.completedSteps).add(step)
  }))
}))
```

### Multi-Tenant State Management

```typescript
interface TenantState {
  currentTenant: Tenant | null
  tenants: Tenant[]
  settings: TenantSettings
  switchTenant: (tenantId: string) => void
}

const useTenantStore = create<TenantState>((set, get) => ({
  currentTenant: null,
  tenants: [],
  settings: {},
  
  switchTenant: async (tenantId) => {
    const tenant = get().tenants.find(t => t.id === tenantId)
    if (tenant) {
      set({ currentTenant: tenant })
      // Clear cache and refetch data for new tenant
      queryClient.resetQueries()
    }
  }
}))
```

---

## Form Handling & Validation

### React Hook Form + Zod Integration

**Benefits for Financial Applications**:
- Type-safe validation schemas
- Complex field relationships
- Currency and percentage formatting
- Cross-field validation (totals, ratios)
- Performance optimization

### Implementation Examples

```typescript
// Zod schema for valuation form
const valuationSchema = z.object({
  companyName: z.string().min(1, 'Company name is required'),
  revenue: z.number().positive('Revenue must be positive'),
  netProfit: z.number(),
  sharesOutstanding: z.number().positive('Shares outstanding must be positive'),
  industry: z.enum(['technology', 'healthcare', 'finance', 'manufacturing']),
  
  // Complex financial validations
  projectedGrowth: z.number()
    .min(0, 'Growth rate cannot be negative')
    .max(10, 'Growth rate seems unrealistic (>1000%)'),
    
  // Cross-field validation
  debtToEquityRatio: z.number()
    .refine((val, ctx) => {
      const equity = ctx.parent.totalEquity
      return equity > 0 && val <= 5
    }, 'Debt-to-equity ratio must be realistic')
})

type ValuationFormData = z.infer<typeof valuationSchema>

// Form component with validation
function ValuationForm() {
  const { register, handleSubmit, formState: { errors }, watch } = useForm<ValuationFormData>({
    resolver: zodResolver(valuationSchema),
    mode: 'onChange'
  })
  
  const revenue = watch('revenue')
  const netProfit = watch('netProfit')
  const profitMargin = revenue > 0 ? (netProfit / revenue) * 100 : 0
  
  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div className="grid grid-cols-2 gap-4">
        <FormField
          label="Annual Revenue (AUD)"
          error={errors.revenue?.message}
        >
          <CurrencyInput
            {...register('revenue', { valueAsNumber: true })}
            placeholder="0.00"
            currency="AUD"
          />
        </FormField>
        
        <FormField
          label="Net Profit (AUD)"
          error={errors.netProfit?.message}
        >
          <CurrencyInput
            {...register('netProfit', { valueAsNumber: true })}
            placeholder="0.00"
            currency="AUD"
          />
        </FormField>
      </div>
      
      {/* Calculated field display */}
      <div className="mt-4 p-4 bg-blue-50 rounded-lg">
        <span className="text-sm font-medium">
          Profit Margin: {profitMargin.toFixed(2)}%
        </span>
      </div>
    </form>
  )
}
```

### File Upload Component

```typescript
interface FileUploadProps {
  accept: string[]
  maxSize: number
  multiple?: boolean
  onUpload: (files: File[]) => Promise<UploadResult[]>
  validationSchema?: z.ZodSchema
}

function DocumentUploader({ accept, maxSize, multiple, onUpload }: FileUploadProps) {
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState<Record<string, number>>({})
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: accept.reduce((acc, type) => ({ ...acc, [type]: [] }), {}),
    maxSize,
    multiple,
    onDrop: async (acceptedFiles) => {
      setUploading(true)
      
      // Upload with progress tracking
      const uploadPromises = acceptedFiles.map(async (file) => {
        const formData = new FormData()
        formData.append('file', file)
        
        return axios.post('/api/upload', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: (event) => {
            const percent = Math.round((event.loaded * 100) / event.total)
            setProgress(prev => ({ ...prev, [file.name]: percent }))
          }
        })
      })
      
      try {
        await Promise.all(uploadPromises)
      } finally {
        setUploading(false)
        setProgress({})
      }
    }
  })
  
  return (
    <div {...getRootProps()} className={cn(
      "border-2 border-dashed rounded-lg p-8 text-center transition-colors",
      isDragActive ? "border-blue-500 bg-blue-50" : "border-gray-300"
    )}>
      <input {...getInputProps()} />
      {/* Upload UI implementation */}
    </div>
  )
}
```

---

## Enterprise Requirements

### Multi-Tenancy Architecture

**Strategy**: Subdomain-based tenant isolation with shared database and application code

```typescript
// Tenant middleware for route handling
import { NextRequest, NextResponse } from 'next/server'

export function middleware(request: NextRequest) {
  const hostname = request.nextUrl.hostname
  const subdomain = hostname.split('.')[0]
  
  // Extract tenant from subdomain
  if (subdomain && subdomain !== 'www' && subdomain !== 'app') {
    // Rewrite to tenant-specific route
    return NextResponse.rewrite(
      new URL(`/(tenant)/${subdomain}${request.nextUrl.pathname}`, request.url)
    )
  }
  
  return NextResponse.next()
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)']
}

// Tenant context provider
interface TenantContextType {
  tenant: Tenant | null
  loading: boolean
  switchTenant: (tenantId: string) => Promise<void>
}

const TenantContext = createContext<TenantContextType | null>(null)

export function TenantProvider({ children }: { children: ReactNode }) {
  const [tenant, setTenant] = useState<Tenant | null>(null)
  const [loading, setLoading] = useState(true)
  
  useEffect(() => {
    // Initialize tenant from subdomain or localStorage
    const initializeTenant = async () => {
      try {
        const tenantData = await fetchTenantFromSubdomain()
        setTenant(tenantData)
      } catch (error) {
        console.error('Failed to initialize tenant:', error)
      } finally {
        setLoading(false)
      }
    }
    
    initializeTenant()
  }, [])
  
  return (
    <TenantContext.Provider value={{ tenant, loading, switchTenant }}>
      {children}
    </TenantContext.Provider>
  )
}
```

### Role-Based Access Control (RBAC)

```typescript
// RBAC types and permissions
type Permission = 
  | 'valuation:read'
  | 'valuation:write'
  | 'reports:read'
  | 'reports:write'
  | 'settings:read'
  | 'settings:write'
  | 'users:manage'

type Role = 'admin' | 'analyst' | 'viewer' | 'client'

const rolePermissions: Record<Role, Permission[]> = {
  admin: ['valuation:read', 'valuation:write', 'reports:read', 'reports:write', 'settings:read', 'settings:write', 'users:manage'],
  analyst: ['valuation:read', 'valuation:write', 'reports:read', 'reports:write'],
  viewer: ['valuation:read', 'reports:read'],
  client: ['reports:read']
}

// HOC for permission-based components
function withPermission<T extends object>(
  Component: React.ComponentType<T>,
  permission: Permission
) {
  return function PermissionGuardedComponent(props: T) {
    const { user } = useAuth()
    const hasPermission = user?.permissions?.includes(permission)
    
    if (!hasPermission) {
      return <AccessDenied requiredPermission={permission} />
    }
    
    return <Component {...props} />
  }
}

// Usage example
const AdminOnlyButton = withPermission(Button, 'settings:write')

// Route-level protection
export default function AdminPage() {
  const { user } = useAuth()
  
  if (!user?.permissions?.includes('users:manage')) {
    return <AccessDenied requiredPermission="users:manage" />
  }
  
  return <AdminDashboard />
}
```

### White-Label Customization

```typescript
// Theme system for white-labeling
interface BrandTheme {
  colors: {
    primary: string
    secondary: string
    accent: string
  }
  fonts: {
    heading: string
    body: string
  }
  logo: string
  companyName: string
}

const useBrandTheme = () => {
  const { tenant } = useTenant()
  
  return useMemo(() => {
    if (tenant?.branding) {
      return {
        colors: {
          primary: tenant.branding.primaryColor,
          secondary: tenant.branding.secondaryColor,
          accent: tenant.branding.accentColor
        },
        fonts: {
          heading: tenant.branding.headingFont,
          body: tenant.branding.bodyFont
        },
        logo: tenant.branding.logoUrl,
        companyName: tenant.branding.companyName
      }
    }
    
    // Default Uprez branding
    return defaultTheme
  }, [tenant])
}

// Dynamic CSS variables
function ThemeProvider({ children }: { children: ReactNode }) {
  const theme = useBrandTheme()
  
  useEffect(() => {
    const root = document.documentElement
    root.style.setProperty('--color-primary', theme.colors.primary)
    root.style.setProperty('--color-secondary', theme.colors.secondary)
    root.style.setProperty('--font-heading', theme.fonts.heading)
  }, [theme])
  
  return <>{children}</>
}
```

### Accessibility (WCAG 2.1 AA Compliance)

```typescript
// Accessibility hooks and utilities
function useA11yAnnouncement() {
  const announce = useCallback((message: string, priority: 'polite' | 'assertive' = 'polite') => {
    const announcer = document.createElement('div')
    announcer.setAttribute('aria-live', priority)
    announcer.setAttribute('aria-atomic', 'true')
    announcer.className = 'sr-only'
    announcer.textContent = message
    
    document.body.appendChild(announcer)
    setTimeout(() => document.body.removeChild(announcer), 1000)
  }, [])
  
  return announce
}

// Focus management for dynamic content
function useFocusManagement() {
  const focusElementRef = useRef<HTMLElement | null>(null)
  
  const setFocusTarget = useCallback((element: HTMLElement | null) => {
    focusElementRef.current = element
  }, [])
  
  const restoreFocus = useCallback(() => {
    if (focusElementRef.current) {
      focusElementRef.current.focus()
      focusElementRef.current = null
    }
  }, [])
  
  return { setFocusTarget, restoreFocus }
}

// Accessible components
function AccessibleButton({ 
  children, 
  onClick, 
  disabled, 
  ariaLabel,
  ariaDescribedBy,
  ...props 
}: ButtonProps) {
  const announce = useA11yAnnouncement()
  
  const handleClick = useCallback((e: MouseEvent<HTMLButtonElement>) => {
    onClick?.(e)
    if (ariaLabel) {
      announce(`${ariaLabel} activated`)
    }
  }, [onClick, ariaLabel, announce])
  
  return (
    <button
      onClick={handleClick}
      disabled={disabled}
      aria-label={ariaLabel}
      aria-describedby={ariaDescribedBy}
      aria-disabled={disabled}
      className={cn(
        "focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500",
        "transition-all duration-200",
        disabled && "opacity-50 cursor-not-allowed"
      )}
      {...props}
    >
      {children}
    </button>
  )
}
```

---

## Performance Optimization

### Core Web Vitals Optimization

**Target Metrics**:
- **LCP (Largest Contentful Paint)**: < 2.5 seconds
- **FID (First Input Delay)**: < 100 milliseconds
- **CLS (Cumulative Layout Shift)**: < 0.1

### Implementation Strategies

```typescript
// Image optimization with Next.js Image component
import Image from 'next/image'

function OptimizedImage({ src, alt, ...props }) {
  return (
    <Image
      src={src}
      alt={alt}
      placeholder="blur"
      blurDataURL="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
      sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
      priority={props.priority}
      {...props}
    />
  )
}

// Bundle optimization with dynamic imports
const HeavyChart = dynamic(() => import('@/components/charts/HeavyChart'), {
  loading: () => <ChartSkeleton />,
  ssr: false // Client-side only for heavy visualizations
})

// Code splitting by route
const AdminPanel = dynamic(() => import('@/components/admin/AdminPanel'), {
  loading: () => <AdminSkeleton />
})

// Prefetching critical resources
function useResourcePrefetch() {
  useEffect(() => {
    // Prefetch critical API endpoints
    const router = useRouter()
    router.prefetch('/api/valuation')
    router.prefetch('/api/reports')
    
    // Preload critical images
    const criticalImages = [
      '/images/dashboard-bg.webp',
      '/images/company-logo.webp'
    ]
    
    criticalImages.forEach(src => {
      const link = document.createElement('link')
      link.rel = 'preload'
      link.as = 'image'
      link.href = src
      document.head.appendChild(link)
    })
  }, [])
}
```

### Data Fetching Optimization

```typescript
// Optimized data fetching patterns
const useOptimizedValuationData = (companyId: string) => {
  return useQuery({
    queryKey: ['valuation', companyId],
    queryFn: async () => {
      // Use Promise.allSettled for parallel requests
      const [metrics, peers, risks] = await Promise.allSettled([
        api.valuation.getMetrics(companyId),
        api.valuation.getPeerAnalysis(companyId),
        api.valuation.getRiskAssessment(companyId)
      ])
      
      return {
        metrics: metrics.status === 'fulfilled' ? metrics.value : null,
        peers: peers.status === 'fulfilled' ? peers.value : null,
        risks: risks.status === 'fulfilled' ? risks.value : null
      }
    },
    staleTime: 5 * 60 * 1000,
    cacheTime: 10 * 60 * 1000,
    select: useCallback((data) => {
      // Transform data on client side for better caching
      return {
        ...data,
        summary: calculateSummaryMetrics(data.metrics)
      }
    }, [])
  })
}

// Infinite scrolling for large datasets
const useInfiniteReports = () => {
  return useInfiniteQuery({
    queryKey: ['reports'],
    queryFn: ({ pageParam = 0 }) => 
      api.reports.getList({ offset: pageParam, limit: 20 }),
    getNextPageParam: (lastPage, pages) => 
      lastPage.hasMore ? pages.length * 20 : undefined
  })
}
```

---

## Security & Compliance

### Authentication & Authorization

```typescript
// JWT token management with secure storage
class TokenManager {
  private static readonly ACCESS_TOKEN_KEY = 'access_token'
  private static readonly REFRESH_TOKEN_KEY = 'refresh_token'
  
  static setTokens(accessToken: string, refreshToken: string) {
    // Use httpOnly cookies for production
    if (typeof window !== 'undefined') {
      document.cookie = `${this.ACCESS_TOKEN_KEY}=${accessToken}; HttpOnly; Secure; SameSite=Strict`
      document.cookie = `${this.REFRESH_TOKEN_KEY}=${refreshToken}; HttpOnly; Secure; SameSite=Strict`
    }
  }
  
  static async refreshToken(): Promise<string | null> {
    try {
      const response = await fetch('/api/auth/refresh', {
        method: 'POST',
        credentials: 'include'
      })
      
      if (response.ok) {
        const { accessToken } = await response.json()
        return accessToken
      }
    } catch (error) {
      console.error('Token refresh failed:', error)
    }
    
    return null
  }
}

// API client with automatic token refresh
class ApiClient {
  private baseURL: string
  
  constructor(baseURL: string) {
    this.baseURL = baseURL
  }
  
  async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseURL}${endpoint}`
    
    const response = await fetch(url, {
      ...options,
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      }
    })
    
    if (response.status === 401) {
      // Attempt token refresh
      const newToken = await TokenManager.refreshToken()
      if (newToken) {
        // Retry request with new token
        return this.request(endpoint, options)
      } else {
        // Redirect to login
        window.location.href = '/login'
        throw new Error('Authentication required')
      }
    }
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`)
    }
    
    return response.json()
  }
}
```

### Data Encryption & Privacy

```typescript
// Client-side encryption for sensitive data
import CryptoJS from 'crypto-js'

class DataEncryption {
  private static readonly SECRET_KEY = process.env.NEXT_PUBLIC_ENCRYPTION_KEY!
  
  static encrypt(data: any): string {
    const jsonString = JSON.stringify(data)
    return CryptoJS.AES.encrypt(jsonString, this.SECRET_KEY).toString()
  }
  
  static decrypt(encryptedData: string): any {
    const bytes = CryptoJS.AES.decrypt(encryptedData, this.SECRET_KEY)
    const jsonString = bytes.toString(CryptoJS.enc.Utf8)
    return JSON.parse(jsonString)
  }
}

// Secure form data handling
function useSecureForm<T extends FieldValues>(schema: ZodSchema<T>) {
  const form = useForm<T>({
    resolver: zodResolver(schema),
    mode: 'onChange'
  })
  
  const secureSubmit = useCallback(async (data: T) => {
    // Encrypt sensitive fields before submission
    const sensitiveFields = ['revenue', 'netProfit', 'valuation']
    const encryptedData = { ...data }
    
    sensitiveFields.forEach(field => {
      if (encryptedData[field]) {
        encryptedData[field] = DataEncryption.encrypt(encryptedData[field])
      }
    })
    
    return api.valuation.submit(encryptedData)
  }, [])
  
  return {
    ...form,
    handleSecureSubmit: form.handleSubmit(secureSubmit)
  }
}
```

### GDPR Compliance

```typescript
// Cookie consent management
interface CookiePreferences {
  necessary: boolean
  analytics: boolean
  marketing: boolean
}

function useCookieConsent() {
  const [preferences, setPreferences] = useState<CookiePreferences>({
    necessary: true, // Always required
    analytics: false,
    marketing: false
  })
  
  const [showBanner, setShowBanner] = useState(false)
  
  useEffect(() => {
    const savedPreferences = localStorage.getItem('cookie_preferences')
    if (savedPreferences) {
      setPreferences(JSON.parse(savedPreferences))
    } else {
      setShowBanner(true)
    }
  }, [])
  
  const updatePreferences = useCallback((newPreferences: CookiePreferences) => {
    setPreferences(newPreferences)
    localStorage.setItem('cookie_preferences', JSON.stringify(newPreferences))
    setShowBanner(false)
    
    // Configure analytics based on preferences
    if (newPreferences.analytics) {
      gtag('consent', 'update', { analytics_storage: 'granted' })
    }
  }, [])
  
  return { preferences, showBanner, updatePreferences }
}

// Data export for GDPR compliance
async function exportUserData(userId: string): Promise<Blob> {
  const userData = await api.users.exportData(userId)
  const jsonString = JSON.stringify(userData, null, 2)
  return new Blob([jsonString], { type: 'application/json' })
}
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
1. **Project Setup**
   - Next.js 15 application initialization
   - TypeScript configuration
   - Tailwind CSS + shadcn/ui setup
   - ESLint and Prettier configuration

2. **Core Architecture**
   - App Router structure implementation
   - Authentication system integration
   - Database setup and ORM configuration
   - API routes foundation

3. **Basic Components**
   - Layout system
   - Navigation components
   - Form components with validation
   - Loading and error states

### Phase 2: Core Features (Weeks 5-8)
1. **Valuation Workflow**
   - Multi-step form implementation
   - File upload system
   - Document processing integration
   - Progress tracking

2. **Data Visualization**
   - Chart component library
   - Real-time data integration
   - Interactive dashboard
   - Export functionality

3. **State Management**
   - Zustand store setup
   - React Query integration
   - Cache management
   - Optimistic updates

### Phase 3: Enterprise Features (Weeks 9-12)
1. **Multi-Tenancy**
   - Tenant isolation
   - Subdomain routing
   - White-label customization
   - Billing integration

2. **Security & Compliance**
   - RBAC implementation
   - Data encryption
   - GDPR compliance
   - Audit logging

3. **Performance Optimization**
   - Code splitting
   - Image optimization
   - Caching strategies
   - Core Web Vitals optimization

### Phase 4: Advanced Features (Weeks 13-16)
1. **Advanced Visualizations**
   - Custom D3.js components
   - Interactive scenarios
   - Advanced filtering
   - Real-time updates

2. **Integration & APIs**
   - Third-party data sources
   - Webhook systems
   - Export capabilities
   - Mobile responsiveness

3. **Testing & Quality Assurance**
   - Unit testing setup
   - Integration testing
   - E2E testing
   - Accessibility testing
   - Performance monitoring

### Success Metrics
- **Performance**: LCP < 2.5s, FID < 100ms, CLS < 0.1
- **Accessibility**: WCAG 2.1 AA compliance score > 95%
- **User Experience**: System Usability Scale (SUS) > 80
- **Security**: Zero critical security vulnerabilities
- **Reliability**: 99.9% uptime, < 1% error rate

---

## Conclusion

This Next.js UI architecture provides a comprehensive foundation for building a world-class IPO valuation SaaS platform. The combination of modern technologies, performance optimization, enterprise-grade security, and excellent user experience positions the platform for success in the competitive financial software market.

The architecture emphasizes:
- **Developer Experience**: TypeScript, modern tooling, clear patterns
- **Performance**: Server-side rendering, streaming, optimized bundles
- **Security**: Multi-layered authentication, encryption, compliance
- **Scalability**: Multi-tenancy, caching, horizontal scaling
- **Accessibility**: WCAG compliance, inclusive design
- **Maintainability**: Clean architecture, comprehensive testing

By following this specification, the development team will deliver a robust, scalable, and user-friendly platform that meets the demanding requirements of financial professionals while providing an exceptional experience for all users.