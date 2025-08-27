import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { User } from '@/types';
import { apiClient } from '@/lib/api';
import { wsClient } from '@/lib/websocket';

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  login: (email: string, password: string) => Promise<void>;
  register: (userData: any) => Promise<void>;
  logout: () => Promise<void>;
  refreshToken: () => Promise<void>;
  updateUser: (userData: Partial<User>) => Promise<void>;
  clearError: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null });
        
        try {
          const response = await apiClient.login(email, password);
          const { token, user } = response.data;
          
          // Update API client token
          apiClient.setToken(token);
          
          // Connect WebSocket
          wsClient.connect(token);
          
          set({
            user,
            token,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error: any) {
          set({
            error: error.message || 'Login failed',
            isLoading: false,
          });
          throw error;
        }
      },

      register: async (userData: any) => {
        set({ isLoading: true, error: null });
        
        try {
          const response = await apiClient.register(userData);
          const { token, user } = response.data;
          
          // Update API client token
          apiClient.setToken(token);
          
          // Connect WebSocket
          wsClient.connect(token);
          
          set({
            user,
            token,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error: any) {
          set({
            error: error.message || 'Registration failed',
            isLoading: false,
          });
          throw error;
        }
      },

      logout: async () => {
        try {
          await apiClient.logout();
        } catch (error) {
          console.error('Logout error:', error);
        } finally {
          // Clear API client token
          apiClient.clearToken();
          
          // Disconnect WebSocket
          wsClient.disconnect();
          
          set({
            user: null,
            token: null,
            isAuthenticated: false,
            error: null,
          });
        }
      },

      refreshToken: async () => {
        const { token } = get();
        if (!token) return;
        
        try {
          const response = await apiClient.refreshToken();
          const newToken = response.data.token;
          
          apiClient.setToken(newToken);
          set({ token: newToken });
        } catch (error) {
          // Token refresh failed, logout
          get().logout();
          throw error;
        }
      },

      updateUser: async (userData: Partial<User>) => {
        set({ isLoading: true, error: null });
        
        try {
          const response = await apiClient.updateUser(userData);
          const updatedUser = response.data;
          
          set({
            user: updatedUser,
            isLoading: false,
          });
        } catch (error: any) {
          set({
            error: error.message || 'Update failed',
            isLoading: false,
          });
          throw error;
        }
      },

      clearError: () => {
        set({ error: null });
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
      onRehydrateStorage: () => (state) => {
        if (state?.token) {
          // Restore API client token
          apiClient.setToken(state.token);
          
          // Reconnect WebSocket
          wsClient.connect(state.token);
        }
      },
    }
  )
);