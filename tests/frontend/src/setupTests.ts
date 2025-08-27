/**
 * Jest setup file for React Testing Library tests.
 * 
 * This file is run before each test file and sets up:
 * - Custom matchers from jest-dom
 * - Mock Service Worker (MSW) for API mocking
 * - Global test utilities and configurations
 * - Canvas and other browser API mocks
 */

import '@testing-library/jest-dom';
import 'jest-canvas-mock';
import { TextEncoder, TextDecoder } from 'util';
import ResizeObserver from 'resize-observer-polyfill';

// Polyfills for Node.js environment
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder as any;

// ResizeObserver polyfill for charts and responsive components
global.ResizeObserver = ResizeObserver;

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  constructor() {}
  observe() {}
  unobserve() {}
  disconnect() {}
};

// Mock matchMedia for responsive design tests
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // deprecated
    removeListener: jest.fn(), // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock window.scrollTo for scroll behavior tests
Object.defineProperty(window, 'scrollTo', {
  writable: true,
  value: jest.fn(),
});

// Mock HTMLElement.scrollIntoView
Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
  writable: true,
  value: jest.fn(),
});

// Mock clipboard API
Object.assign(navigator, {
  clipboard: {
    writeText: jest.fn(),
    readText: jest.fn(),
  },
});

// Mock File and FileReader for file upload tests
global.File = class MockFile {
  constructor(
    public bits: BlobPart[],
    public name: string,
    public options?: FilePropertyBag
  ) {}
  get size() {
    return this.bits.reduce((size, bit) => {
      if (typeof bit === 'string') return size + bit.length;
      if (bit instanceof ArrayBuffer) return size + bit.byteLength;
      return size + (bit as any).length || 0;
    }, 0);
  }
  get type() {
    return this.options?.type || '';
  }
} as any;

global.FileReader = class MockFileReader {
  public readyState = 0;
  public result: string | ArrayBuffer | null = null;
  public error: any = null;
  public onload: ((event: any) => void) | null = null;
  public onerror: ((event: any) => void) | null = null;
  public onloadend: ((event: any) => void) | null = null;

  readAsDataURL(file: File) {
    setTimeout(() => {
      this.result = `data:${file.type};base64,mock-base64-content`;
      this.readyState = 2;
      if (this.onload) {
        this.onload({ target: this } as any);
      }
      if (this.onloadend) {
        this.onloadend({ target: this } as any);
      }
    }, 0);
  }

  readAsText(file: File) {
    setTimeout(() => {
      this.result = 'mock file content';
      this.readyState = 2;
      if (this.onload) {
        this.onload({ target: this } as any);
      }
      if (this.onloadend) {
        this.onloadend({ target: this } as any);
      }
    }, 0);
  }
} as any;

// Mock URL.createObjectURL for file preview tests
global.URL.createObjectURL = jest.fn((object: File | Blob) => {
  return `mock-object-url-${object.constructor.name}`;
});

global.URL.revokeObjectURL = jest.fn();

// Mock console methods to reduce noise in tests
const originalConsoleError = console.error;
const originalConsoleWarn = console.warn;

beforeAll(() => {
  console.error = (...args: any[]) => {
    // Suppress React 18 strict mode warnings in tests
    if (args[0]?.includes?.('Warning:')) return;
    originalConsoleError.call(console, ...args);
  };

  console.warn = (...args: any[]) => {
    // Suppress development warnings
    if (args[0]?.includes?.('Warning:')) return;
    originalConsoleWarn.call(console, ...args);
  };
});

afterAll(() => {
  console.error = originalConsoleError;
  console.warn = originalConsoleWarn;
});

// Clean up after each test
afterEach(() => {
  // Clear all mocks
  jest.clearAllMocks();
  
  // Clean up any DOM modifications
  document.body.innerHTML = '';
  
  // Reset any global state
  localStorage.clear();
  sessionStorage.clear();
});

// Global test timeout
jest.setTimeout(10000);

// Mock environment variables for tests
process.env.NODE_ENV = 'test';
process.env.REACT_APP_API_BASE_URL = 'http://localhost:8000/api/v1';
process.env.REACT_APP_ENVIRONMENT = 'test';