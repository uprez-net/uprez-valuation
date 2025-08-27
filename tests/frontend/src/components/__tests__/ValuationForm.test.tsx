/**
 * Unit tests for ValuationForm component.
 * 
 * Tests form validation, user interactions, API integration,
 * and edge cases for the main valuation request form.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { rest } from 'msw';
import { setupServer } from 'msw/node';

import ValuationForm from '../ValuationForm';
import { ApiError, ValidationError } from '../../types/errors';

// Mock API server
const server = setupServer(
  rest.post('/api/v1/valuations', (req, res, ctx) => {
    return res(
      ctx.status(201),
      ctx.json({
        job_id: 'val_mock_job_123456789',
        status: 'queued',
        estimated_completion_time: '12 minutes',
        created_at: new Date().toISOString(),
        polling_url: '/api/v1/valuations/val_mock_job_123456789/status'
      })
    );
  }),

  rest.get('/api/v1/industries', (req, res, ctx) => {
    return res(
      ctx.json([
        { value: 'technology', label: 'Technology' },
        { value: 'healthcare', label: 'Healthcare' },
        { value: 'financial-services', label: 'Financial Services' },
        { value: 'materials', label: 'Materials' }
      ])
    );
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// Test wrapper with providers
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
      mutations: {
        retry: false,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <BrowserRouter>
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    </BrowserRouter>
  );
};

describe('ValuationForm Component', () => {
  beforeEach(() => {
    // Reset localStorage before each test
    localStorage.clear();
  });

  describe('Form Rendering', () => {
    it('renders all required form fields', async () => {
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      // Company Information Section
      expect(screen.getByLabelText(/company name/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/australian business number/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/industry sector/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/business description/i)).toBeInTheDocument();

      // IPO Details Section
      expect(screen.getByLabelText(/expected ipo date/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/shares to be issued/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/minimum price/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/maximum price/i)).toBeInTheDocument();

      // Analysis Options Section
      expect(screen.getByLabelText(/analysis type/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/include peer analysis/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/risk assessment level/i)).toBeInTheDocument();

      // Submit button
      expect(screen.getByRole('button', { name: /create valuation/i })).toBeInTheDocument();
    });

    it('renders with correct initial values', () => {
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      // Check default values
      expect(screen.getByDisplayValue('comprehensive')).toBeInTheDocument(); // Analysis type
      expect(screen.getByRole('checkbox', { name: /include peer analysis/i })).toBeChecked();
      expect(screen.getByDisplayValue('detailed')).toBeInTheDocument(); // Risk assessment level
    });

    it('displays loading state while fetching industries', async () => {
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      // Should show loading state for industry dropdown
      const industrySelect = screen.getByLabelText(/industry sector/i);
      expect(industrySelect).toBeDisabled();

      // Wait for industries to load
      await waitFor(() => {
        expect(industrySelect).toBeEnabled();
      });
    });
  });

  describe('Form Validation', () => {
    it('validates required fields', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      // Try to submit empty form
      const submitButton = screen.getByRole('button', { name: /create valuation/i });
      await user.click(submitButton);

      // Should show validation errors
      await waitFor(() => {
        expect(screen.getByText(/company name is required/i)).toBeInTheDocument();
        expect(screen.getByText(/industry sector is required/i)).toBeInTheDocument();
        expect(screen.getByText(/expected ipo date is required/i)).toBeInTheDocument();
      });
    });

    it('validates ABN format', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      const abnField = screen.getByLabelText(/australian business number/i);
      await user.type(abnField, '123'); // Invalid ABN

      await user.tab(); // Trigger blur validation

      await waitFor(() => {
        expect(screen.getByText(/invalid abn format/i)).toBeInTheDocument();
      });
    });

    it('validates price range logic', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      const minPriceField = screen.getByLabelText(/minimum price/i);
      const maxPriceField = screen.getByLabelText(/maximum price/i);

      await user.type(minPriceField, '5.00');
      await user.type(maxPriceField, '3.00'); // Max less than min

      await user.tab();

      await waitFor(() => {
        expect(screen.getByText(/maximum price must be greater than minimum price/i)).toBeInTheDocument();
      });
    });

    it('validates future IPO date', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      const ipoDateField = screen.getByLabelText(/expected ipo date/i);
      
      // Set date to yesterday
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      const yesterdayString = yesterday.toISOString().split('T')[0];

      await user.type(ipoDateField, yesterdayString);
      await user.tab();

      await waitFor(() => {
        expect(screen.getByText(/ipo date must be in the future/i)).toBeInTheDocument();
      });
    });

    it('validates shares outstanding range', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      const sharesField = screen.getByLabelText(/shares to be issued/i);
      
      await user.type(sharesField, '500'); // Too few shares
      await user.tab();

      await waitFor(() => {
        expect(screen.getByText(/minimum 1,000 shares required/i)).toBeInTheDocument();
      });

      await user.clear(sharesField);
      await user.type(sharesField, '10000000000'); // Too many shares
      await user.tab();

      await waitFor(() => {
        expect(screen.getByText(/maximum 5 billion shares allowed/i)).toBeInTheDocument();
      });
    });

    it('validates business description length', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      const descriptionField = screen.getByLabelText(/business description/i);
      const longDescription = 'A'.repeat(1001); // Over 1000 character limit

      await user.type(descriptionField, longDescription);
      await user.tab();

      await waitFor(() => {
        expect(screen.getByText(/description must be less than 1000 characters/i)).toBeInTheDocument();
      });
    });
  });

  describe('Form Interaction', () => {
    it('handles form field changes correctly', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      // Fill out company name
      const companyNameField = screen.getByLabelText(/company name/i);
      await user.type(companyNameField, 'Test Company Pty Ltd');
      expect(companyNameField).toHaveValue('Test Company Pty Ltd');

      // Select analysis type
      const analysisTypeSelect = screen.getByLabelText(/analysis type/i);
      await user.selectOptions(analysisTypeSelect, 'premium');
      expect(screen.getByDisplayValue('premium')).toBeInTheDocument();

      // Toggle peer analysis checkbox
      const peerAnalysisCheckbox = screen.getByLabelText(/include peer analysis/i);
      await user.click(peerAnalysisCheckbox);
      expect(peerAnalysisCheckbox).not.toBeChecked();
    });

    it('shows/hides conditional fields based on selections', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      // Premium analysis should show additional options
      const analysisTypeSelect = screen.getByLabelText(/analysis type/i);
      await user.selectOptions(analysisTypeSelect, 'premium');

      await waitFor(() => {
        expect(screen.getByLabelText(/custom parameters/i)).toBeInTheDocument();
      });

      // Basic analysis should hide additional options
      await user.selectOptions(analysisTypeSelect, 'basic');

      await waitFor(() => {
        expect(screen.queryByLabelText(/custom parameters/i)).not.toBeInTheDocument();
      });
    });

    it('auto-formats currency inputs', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      const minPriceField = screen.getByLabelText(/minimum price/i);
      await user.type(minPriceField, '2.5');
      
      // Should format to 2 decimal places
      expect(minPriceField).toHaveValue('2.50');
    });

    it('auto-formats number inputs with thousands separators', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      const sharesField = screen.getByLabelText(/shares to be issued/i);
      await user.type(sharesField, '50000000');
      
      // Should format with commas
      expect(sharesField).toHaveValue('50,000,000');
    });

    it('saves form data to localStorage as user types', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      const companyNameField = screen.getByLabelText(/company name/i);
      await user.type(companyNameField, 'Draft Company');

      // Check that data is saved to localStorage
      await waitFor(() => {
        const savedData = JSON.parse(localStorage.getItem('valuation-form-draft') || '{}');
        expect(savedData.companyName).toBe('Draft Company');
      });
    });

    it('restores form data from localStorage on mount', () => {
      const draftData = {
        companyName: 'Restored Company',
        industryType: 'technology',
        sharesOutstanding: 25000000
      };
      localStorage.setItem('valuation-form-draft', JSON.stringify(draftData));

      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      expect(screen.getByDisplayValue('Restored Company')).toBeInTheDocument();
      expect(screen.getByDisplayValue('25,000,000')).toBeInTheDocument();
    });
  });

  describe('Form Submission', () => {
    it('submits form with valid data successfully', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      // Fill out required fields
      await user.type(screen.getByLabelText(/company name/i), 'TechCorp Pty Ltd');
      await user.type(screen.getByLabelText(/australian business number/i), '12345678901');
      
      // Wait for industries to load and select one
      await waitFor(() => {
        expect(screen.getByLabelText(/industry sector/i)).toBeEnabled();
      });
      await user.selectOptions(screen.getByLabelText(/industry sector/i), 'technology');
      
      await user.type(screen.getByLabelText(/business description/i), 'AI-powered fintech platform');
      
      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 30);
      const tomorrowString = tomorrow.toISOString().split('T')[0];
      await user.type(screen.getByLabelText(/expected ipo date/i), tomorrowString);
      
      await user.type(screen.getByLabelText(/shares to be issued/i), '50000000');
      await user.type(screen.getByLabelText(/minimum price/i), '2.50');
      await user.type(screen.getByLabelText(/maximum price/i), '4.00');

      // Submit form
      const submitButton = screen.getByRole('button', { name: /create valuation/i });
      await user.click(submitButton);

      // Should show loading state
      await waitFor(() => {
        expect(screen.getByRole('button', { name: /creating.../i })).toBeInTheDocument();
      });

      // Should show success message
      await waitFor(() => {
        expect(screen.getByText(/valuation request created successfully/i)).toBeInTheDocument();
        expect(screen.getByText(/job id: val_mock_job_123456789/i)).toBeInTheDocument();
      });

      // Should clear localStorage draft
      expect(localStorage.getItem('valuation-form-draft')).toBeNull();
    });

    it('handles API validation errors', async () => {
      const user = userEvent.setup();
      
      // Mock API error response
      server.use(
        rest.post('/api/v1/valuations', (req, res, ctx) => {
          return res(
            ctx.status(400),
            ctx.json({
              error: 'validation_error',
              message: 'The request contains invalid parameters',
              details: {
                field_errors: {
                  expected_ipo_date: 'Date must be at least 30 days in the future',
                  shares_on_issue: 'Must be between 1,000 and 1,000,000,000'
                }
              }
            })
          );
        })
      );

      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      // Fill out form with data that will trigger server validation error
      await user.type(screen.getByLabelText(/company name/i), 'TechCorp Pty Ltd');
      await user.type(screen.getByLabelText(/australian business number/i), '12345678901');
      
      await waitFor(() => {
        expect(screen.getByLabelText(/industry sector/i)).toBeEnabled();
      });
      await user.selectOptions(screen.getByLabelText(/industry sector/i), 'technology');
      
      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 1);
      const tomorrowString = tomorrow.toISOString().split('T')[0];
      await user.type(screen.getByLabelText(/expected ipo date/i), tomorrowString);
      
      await user.type(screen.getByLabelText(/shares to be issued/i), '500');

      const submitButton = screen.getByRole('button', { name: /create valuation/i });
      await user.click(submitButton);

      // Should show field-specific errors
      await waitFor(() => {
        expect(screen.getByText(/date must be at least 30 days in the future/i)).toBeInTheDocument();
        expect(screen.getByText(/must be between 1,000 and 1,000,000,000/i)).toBeInTheDocument();
      });

      // Submit button should be enabled again
      expect(screen.getByRole('button', { name: /create valuation/i })).toBeEnabled();
    });

    it('handles network errors gracefully', async () => {
      const user = userEvent.setup();
      
      // Mock network error
      server.use(
        rest.post('/api/v1/valuations', (req, res, ctx) => {
          return res.networkError('Failed to connect');
        })
      );

      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      // Fill out minimal valid form
      await user.type(screen.getByLabelText(/company name/i), 'TechCorp Pty Ltd');
      await waitFor(() => {
        expect(screen.getByLabelText(/industry sector/i)).toBeEnabled();
      });
      await user.selectOptions(screen.getByLabelText(/industry sector/i), 'technology');
      
      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 30);
      const tomorrowString = tomorrow.toISOString().split('T')[0];
      await user.type(screen.getByLabelText(/expected ipo date/i), tomorrowString);

      const submitButton = screen.getByRole('button', { name: /create valuation/i });
      await user.click(submitButton);

      // Should show network error message
      await waitFor(() => {
        expect(screen.getByText(/network error occurred/i)).toBeInTheDocument();
        expect(screen.getByText(/please check your connection and try again/i)).toBeInTheDocument();
      });

      // Should have retry button
      expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
    });

    it('handles server errors gracefully', async () => {
      const user = userEvent.setup();
      
      // Mock server error
      server.use(
        rest.post('/api/v1/valuations', (req, res, ctx) => {
          return res(
            ctx.status(500),
            ctx.json({
              error: 'internal_server_error',
              message: 'An unexpected error occurred'
            })
          );
        })
      );

      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      // Fill out minimal valid form
      await user.type(screen.getByLabelText(/company name/i), 'TechCorp Pty Ltd');
      await waitFor(() => {
        expect(screen.getByLabelText(/industry sector/i)).toBeEnabled();
      });
      await user.selectOptions(screen.getByLabelText(/industry sector/i), 'technology');
      
      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 30);
      const tomorrowString = tomorrow.toISOString().split('T')[0];
      await user.type(screen.getByLabelText(/expected ipo date/i), tomorrowString);

      const submitButton = screen.getByRole('button', { name: /create valuation/i });
      await user.click(submitButton);

      // Should show generic error message
      await waitFor(() => {
        expect(screen.getByText(/an unexpected error occurred/i)).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels and descriptions', () => {
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      // Check form has proper labeling
      expect(screen.getByRole('form')).toHaveAttribute('aria-label', 'Create New IPO Valuation');

      // Check required fields are marked
      const requiredFields = screen.getAllByLabelText(/\*/);
      expect(requiredFields.length).toBeGreaterThan(0);

      // Check error messages are associated with fields
      const companyNameField = screen.getByLabelText(/company name/i);
      expect(companyNameField).toHaveAttribute('aria-describedby');
    });

    it('has proper focus management', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      // Tab through form elements
      await user.tab();
      expect(screen.getByLabelText(/company name/i)).toHaveFocus();

      await user.tab();
      expect(screen.getByLabelText(/australian business number/i)).toHaveFocus();
    });

    it('announces validation errors to screen readers', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      // Trigger validation error
      const submitButton = screen.getByRole('button', { name: /create valuation/i });
      await user.click(submitButton);

      // Check that error messages have proper ARIA attributes
      await waitFor(() => {
        const errorMessage = screen.getByText(/company name is required/i);
        expect(errorMessage).toHaveAttribute('role', 'alert');
      });
    });
  });

  describe('Responsive Behavior', () => {
    it('adjusts layout for mobile screens', () => {
      // Mock mobile viewport
      Object.defineProperty(window, 'matchMedia', {
        writable: true,
        value: jest.fn().mockImplementation((query) => ({
          matches: query === '(max-width: 768px)',
          media: query,
          onchange: null,
          addListener: jest.fn(),
          removeListener: jest.fn(),
          addEventListener: jest.fn(),
          removeEventListener: jest.fn(),
          dispatchEvent: jest.fn(),
        })),
      });

      const Wrapper = createWrapper();
      render(<ValuationForm />, { wrapper: Wrapper });

      // Check that mobile-specific classes are applied
      const form = screen.getByRole('form');
      expect(form).toHaveClass('mobile-layout');
    });
  });

  describe('Performance', () => {
    it('debounces localStorage saves', async () => {
      const user = userEvent.setup();
      const Wrapper = createWrapper();
      const setItemSpy = jest.spyOn(Storage.prototype, 'setItem');
      
      render(<ValuationForm />, { wrapper: Wrapper });

      const companyNameField = screen.getByLabelText(/company name/i);
      
      // Type multiple characters quickly
      await user.type(companyNameField, 'Test');

      // Should debounce and only save once
      await waitFor(() => {
        expect(setItemSpy).toHaveBeenCalledTimes(1);
      }, { timeout: 1000 });

      setItemSpy.mockRestore();
    });

    it('memoizes expensive calculations', () => {
      const Wrapper = createWrapper();
      const { rerender } = render(<ValuationForm />, { wrapper: Wrapper });

      // Re-render with same props should not recalculate
      rerender(<ValuationForm />);

      // Verify memoization by checking that expensive operations aren't repeated
      // This would be implementation-specific based on actual component
    });
  });
});