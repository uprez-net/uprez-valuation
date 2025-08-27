import { APIResponse, APIError, PaginatedResponse } from '@/types';

class APIClient {
  private baseURL: string;
  private token: string | null = null;

  constructor() {
    this.baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  }

  setToken(token: string) {
    this.token = token;
  }

  clearToken() {
    this.token = null;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...(this.token && { Authorization: `Bearer ${this.token}` }),
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new APIError(
          errorData.code || `HTTP_${response.status}`,
          errorData.message || response.statusText,
          errorData.details
        );
      }

      const data = await response.json();
      return data;
    } catch (error) {
      if (error instanceof APIError) {
        throw error;
      }
      
      // Network or parsing error
      throw new APIError(
        'NETWORK_ERROR',
        'Failed to connect to the server',
        { originalError: error }
      );
    }
  }

  // Authentication
  async login(email: string, password: string) {
    return this.request<APIResponse<{ token: string; user: any }>>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  }

  async register(userData: any) {
    return this.request<APIResponse<{ token: string; user: any }>>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  }

  async refreshToken() {
    return this.request<APIResponse<{ token: string }>>('/auth/refresh', {
      method: 'POST',
    });
  }

  async logout() {
    return this.request<APIResponse<{}>>('/auth/logout', {
      method: 'POST',
    });
  }

  // Users
  async getCurrentUser() {
    return this.request<APIResponse<any>>('/users/me');
  }

  async updateUser(userData: any) {
    return this.request<APIResponse<any>>('/users/me', {
      method: 'PUT',
      body: JSON.stringify(userData),
    });
  }

  // Projects
  async getProjects(params?: { page?: number; limit?: number; status?: string }) {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString());
        }
      });
    }
    
    return this.request<PaginatedResponse<any>>(`/projects?${searchParams}`);
  }

  async getProject(id: string) {
    return this.request<APIResponse<any>>(`/projects/${id}`);
  }

  async createProject(projectData: any) {
    return this.request<APIResponse<any>>('/projects', {
      method: 'POST',
      body: JSON.stringify(projectData),
    });
  }

  async updateProject(id: string, projectData: any) {
    return this.request<APIResponse<any>>(`/projects/${id}`, {
      method: 'PUT',
      body: JSON.stringify(projectData),
    });
  }

  async deleteProject(id: string) {
    return this.request<APIResponse<{}>>(`/projects/${id}`, {
      method: 'DELETE',
    });
  }

  // Documents
  async uploadDocument(projectId: string, file: File, onProgress?: (progress: number) => void) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('projectId', projectId);

    return new Promise<APIResponse<any>>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable && onProgress) {
          const progress = (e.loaded / e.total) * 100;
          onProgress(progress);
        }
      });
      
      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const response = JSON.parse(xhr.responseText);
            resolve(response);
          } catch (error) {
            reject(new APIError('PARSE_ERROR', 'Failed to parse response'));
          }
        } else {
          try {
            const errorData = JSON.parse(xhr.responseText);
            reject(new APIError(
              errorData.code || `HTTP_${xhr.status}`,
              errorData.message || xhr.statusText
            ));
          } catch {
            reject(new APIError(`HTTP_${xhr.status}`, xhr.statusText));
          }
        }
      });
      
      xhr.addEventListener('error', () => {
        reject(new APIError('NETWORK_ERROR', 'Upload failed'));
      });
      
      xhr.open('POST', `${this.baseURL}/documents/upload`);
      if (this.token) {
        xhr.setRequestHeader('Authorization', `Bearer ${this.token}`);
      }
      xhr.send(formData);
    });
  }

  async getDocuments(projectId: string) {
    return this.request<APIResponse<any[]>>(`/projects/${projectId}/documents`);
  }

  async deleteDocument(id: string) {
    return this.request<APIResponse<{}>>(`/documents/${id}`, {
      method: 'DELETE',
    });
  }

  // Valuation
  async runValuation(projectId: string, scenarios: any[]) {
    return this.request<APIResponse<any>>(`/projects/${projectId}/valuation`, {
      method: 'POST',
      body: JSON.stringify({ scenarios }),
    });
  }

  async getValuationResults(projectId: string) {
    return this.request<APIResponse<any>>(`/projects/${projectId}/valuation`);
  }

  // Reports
  async generateReport(projectId: string, format: 'pdf' | 'excel') {
    const response = await fetch(`${this.baseURL}/projects/${projectId}/report?format=${format}`, {
      headers: {
        ...(this.token && { Authorization: `Bearer ${this.token}` }),
      },
    });

    if (!response.ok) {
      throw new APIError(`HTTP_${response.status}`, response.statusText);
    }

    return response.blob();
  }

  // Analytics
  async getAnalytics(params?: { startDate?: string; endDate?: string; projectId?: string }) {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value);
        }
      });
    }
    
    return this.request<APIResponse<any>>(`/analytics?${searchParams}`);
  }
}

export const apiClient = new APIClient();