import { z } from 'zod';

// Authentication Schemas
export const loginSchema = z.object({
  email: z.string().email('Please enter a valid email address'),
  password: z.string().min(1, 'Password is required'),
  rememberMe: z.boolean().optional().default(false),
});

export const registerSchema = z.object({
  name: z.string().min(2, 'Name must be at least 2 characters'),
  email: z.string().email('Please enter a valid email address'),
  password: z.string()
    .min(8, 'Password must be at least 8 characters')
    .regex(
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
      'Password must contain at least one uppercase letter, one lowercase letter, and one number'
    ),
  confirmPassword: z.string(),
  organizationName: z.string().min(2, 'Organization name must be at least 2 characters'),
  agreeToTerms: z.boolean().refine(val => val === true, 'You must agree to the terms and conditions'),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

export const forgotPasswordSchema = z.object({
  email: z.string().email('Please enter a valid email address'),
});

export const resetPasswordSchema = z.object({
  password: z.string()
    .min(8, 'Password must be at least 8 characters')
    .regex(
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
      'Password must contain at least one uppercase letter, one lowercase letter, and one number'
    ),
  confirmPassword: z.string(),
  token: z.string().min(1, 'Reset token is required'),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

// Project Schemas
export const projectSchema = z.object({
  name: z.string().min(3, 'Project name must be at least 3 characters').max(100, 'Project name must not exceed 100 characters'),
  company: z.string().min(2, 'Company name must be at least 2 characters').max(100, 'Company name must not exceed 100 characters'),
  description: z.string().max(500, 'Description must not exceed 500 characters').optional(),
  dueDate: z.string().optional().refine((date) => {
    if (!date) return true;
    return new Date(date) > new Date();
  }, 'Due date must be in the future'),
  assignedTo: z.array(z.string()).optional(),
  industry: z.string().optional(),
  stage: z.enum(['pre-ipo', 'ipo-filed', 'ipo-priced', 'post-ipo']).optional(),
});

// Valuation Schemas
export const scenarioSchema = z.object({
  name: z.string().min(1, 'Scenario name is required').max(50, 'Scenario name must not exceed 50 characters'),
  type: z.enum(['base', 'bull', 'bear', 'custom']),
  assumptions: z.object({
    revenueGrowth: z.number().min(-1, 'Revenue growth cannot be less than -100%').max(10, 'Revenue growth cannot exceed 1000%'),
    ebitdaMargin: z.number().min(0, 'EBITDA margin cannot be negative').max(1, 'EBITDA margin cannot exceed 100%'),
    discountRate: z.number().min(0.01, 'Discount rate must be at least 0.01%').max(0.5, 'Discount rate cannot exceed 50%'),
    terminalGrowthRate: z.number().min(0, 'Terminal growth rate cannot be negative').max(0.1, 'Terminal growth rate cannot exceed 10%'),
    multipleRange: z.object({
      min: z.number().min(0, 'Minimum multiple cannot be negative'),
      max: z.number().min(0, 'Maximum multiple cannot be negative'),
    }),
  }),
});

export const bulkScenariosSchema = z.object({
  scenarios: z.array(scenarioSchema).min(1, 'At least one scenario is required').max(10, 'Cannot create more than 10 scenarios at once'),
});

// Document Upload Schemas
export const documentUploadSchema = z.object({
  projectId: z.string().min(1, 'Project ID is required'),
  type: z.enum(['prospectus', 'financial_statement', 'research_report', 'other']),
  description: z.string().max(200, 'Description must not exceed 200 characters').optional(),
});

// User Profile Schemas
export const updateProfileSchema = z.object({
  name: z.string().min(2, 'Name must be at least 2 characters').max(100, 'Name must not exceed 100 characters'),
  email: z.string().email('Please enter a valid email address'),
  phone: z.string().regex(/^\+?[1-9]\d{1,14}$/, 'Please enter a valid phone number').optional(),
  timezone: z.string().optional(),
  notifications: z.object({
    email: z.boolean().default(true),
    push: z.boolean().default(true),
    desktop: z.boolean().default(false),
  }).optional(),
});

export const changePasswordSchema = z.object({
  currentPassword: z.string().min(1, 'Current password is required'),
  newPassword: z.string()
    .min(8, 'Password must be at least 8 characters')
    .regex(
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
      'Password must contain at least one uppercase letter, one lowercase letter, and one number'
    ),
  confirmNewPassword: z.string(),
}).refine((data) => data.newPassword === data.confirmNewPassword, {
  message: "New passwords don't match",
  path: ["confirmNewPassword"],
});

// Organization Schemas
export const organizationSchema = z.object({
  name: z.string().min(2, 'Organization name must be at least 2 characters').max(100, 'Organization name must not exceed 100 characters'),
  domain: z.string().regex(/^[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}$/, 'Please enter a valid domain').optional(),
  industry: z.string().max(50, 'Industry must not exceed 50 characters').optional(),
  size: z.enum(['1-10', '11-50', '51-200', '201-1000', '1000+']).optional(),
  branding: z.object({
    logo: z.string().url('Logo must be a valid URL').optional(),
    primaryColor: z.string().regex(/^#[0-9A-F]{6}$/i, 'Primary color must be a valid hex color'),
    secondaryColor: z.string().regex(/^#[0-9A-F]{6}$/i, 'Secondary color must be a valid hex color'),
  }).optional(),
});

// Search and Filter Schemas
export const searchSchema = z.object({
  query: z.string().max(200, 'Search query must not exceed 200 characters').optional(),
  filters: z.object({
    status: z.array(z.enum(['draft', 'in_progress', 'completed', 'archived'])).optional(),
    dateRange: z.object({
      start: z.string().optional(),
      end: z.string().optional(),
    }).optional(),
    assignedTo: z.array(z.string()).optional(),
    industry: z.array(z.string()).optional(),
    valuationRange: z.object({
      min: z.number().optional(),
      max: z.number().optional(),
    }).optional(),
  }).optional(),
  sort: z.object({
    field: z.enum(['name', 'createdAt', 'updatedAt', 'dueDate', 'value']),
    direction: z.enum(['asc', 'desc']),
  }).optional(),
  page: z.number().min(1).default(1),
  limit: z.number().min(1).max(100).default(20),
});

// Report Generation Schemas
export const reportConfigSchema = z.object({
  projectId: z.string().min(1, 'Project ID is required'),
  format: z.enum(['pdf', 'excel', 'word']),
  sections: z.array(z.enum([
    'executive_summary',
    'company_overview',
    'financial_analysis',
    'valuation_results',
    'scenarios_comparison',
    'risk_analysis',
    'appendices'
  ])).min(1, 'At least one section must be selected'),
  includeCharts: z.boolean().default(true),
  includeTables: z.boolean().default(true),
  customization: z.object({
    logo: z.string().url().optional(),
    companyName: z.string().optional(),
    footerText: z.string().max(100).optional(),
  }).optional(),
});

// API Response Schemas
export const paginationSchema = z.object({
  page: z.number().min(1),
  limit: z.number().min(1).max(100),
  total: z.number().min(0),
  totalPages: z.number().min(0),
});

export const apiResponseSchema = <T>(dataSchema: z.ZodSchema<T>) => z.object({
  data: dataSchema,
  message: z.string(),
  success: z.boolean(),
  timestamp: z.string(),
});

export const paginatedResponseSchema = <T>(dataSchema: z.ZodSchema<T>) => z.object({
  data: z.array(dataSchema),
  pagination: paginationSchema,
  message: z.string().optional(),
  success: z.boolean().default(true),
});

// Type exports
export type LoginFormData = z.infer<typeof loginSchema>;
export type RegisterFormData = z.infer<typeof registerSchema>;
export type ForgotPasswordFormData = z.infer<typeof forgotPasswordSchema>;
export type ResetPasswordFormData = z.infer<typeof resetPasswordSchema>;
export type ProjectFormData = z.infer<typeof projectSchema>;
export type ScenarioFormData = z.infer<typeof scenarioSchema>;
export type BulkScenariosFormData = z.infer<typeof bulkScenariosSchema>;
export type DocumentUploadFormData = z.infer<typeof documentUploadSchema>;
export type UpdateProfileFormData = z.infer<typeof updateProfileSchema>;
export type ChangePasswordFormData = z.infer<typeof changePasswordSchema>;
export type OrganizationFormData = z.infer<typeof organizationSchema>;
export type SearchFormData = z.infer<typeof searchSchema>;
export type ReportConfigFormData = z.infer<typeof reportConfigSchema>;