// Core Types
export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'analyst' | 'viewer';
  organization?: Organization;
  avatar?: string;
  createdAt: string;
  lastLogin?: string;
}

export interface Organization {
  id: string;
  name: string;
  domain: string;
  branding: {
    logo?: string;
    primaryColor: string;
    secondaryColor: string;
  };
  plan: 'starter' | 'professional' | 'enterprise';
  createdAt: string;
}

// Valuation Types
export interface ValuationProject {
  id: string;
  name: string;
  company: string;
  status: 'draft' | 'in_progress' | 'completed' | 'archived';
  createdBy: User;
  assignedTo?: User[];
  metrics: ValuationMetrics;
  scenarios: ValuationScenario[];
  documents: Document[];
  createdAt: string;
  updatedAt: string;
  dueDate?: string;
}

export interface ValuationMetrics {
  revenue: number;
  ebitda: number;
  netIncome: number;
  totalAssets: number;
  totalLiabilities: number;
  sharesOutstanding: number;
  bookValuePerShare: number;
  priceToBookRatio: number;
  priceToEarningsRatio: number;
  enterpriseValue: number;
  marketCap: number;
  impliedValuation: number;
}

export interface ValuationScenario {
  id: string;
  name: string;
  type: 'base' | 'bull' | 'bear' | 'custom';
  assumptions: {
    revenueGrowth: number;
    ebitdaMargin: number;
    discountRate: number;
    terminalGrowthRate: number;
    multipleRange: {
      min: number;
      max: number;
    };
  };
  results: {
    dcfValue: number;
    comparableValue: number;
    impliedValue: number;
    confidence: number;
  };
}

export interface Document {
  id: string;
  name: string;
  type: 'prospectus' | 'financial_statement' | 'research_report' | 'other';
  status: 'uploaded' | 'processing' | 'processed' | 'error';
  url: string;
  size: number;
  uploadedBy: User;
  uploadedAt: string;
  extractedData?: any;
}

// Chart Types
export interface ChartDataPoint {
  label: string;
  value: number;
  date?: string;
  category?: string;
}

export interface WaterfallDataPoint {
  name: string;
  value: number;
  cumulative: number;
  type: 'positive' | 'negative' | 'total';
}

// API Types
export interface APIResponse<T> {
  data: T;
  message: string;
  success: boolean;
  timestamp: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

export interface APIError {
  code: string;
  message: string;
  details?: any;
}

// Form Types
export interface LoginForm {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterForm {
  email: string;
  password: string;
  confirmPassword: string;
  name: string;
  organizationName?: string;
}

export interface ProjectForm {
  name: string;
  company: string;
  description?: string;
  dueDate?: string;
  assignedTo?: string[];
}

// WebSocket Types
export interface WebSocketMessage {
  type: 'project_update' | 'user_joined' | 'user_left' | 'document_processed' | 'calculation_complete';
  data: any;
  timestamp: string;
  userId: string;
}

// Theme Types
export interface Theme {
  name: string;
  colors: {
    primary: string;
    secondary: string;
    background: string;
    foreground: string;
    muted: string;
    accent: string;
  };
}

// Upload Types
export interface FileUploadProgress {
  file: File;
  progress: number;
  status: 'pending' | 'uploading' | 'completed' | 'error';
  error?: string;
}