import { create } from 'zustand';
import { ValuationProject, Document, ValuationScenario } from '@/types';
import { apiClient } from '@/lib/api';

interface ProjectsState {
  projects: ValuationProject[];
  currentProject: ValuationProject | null;
  isLoading: boolean;
  error: string | null;
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
  
  // Actions
  fetchProjects: (params?: { page?: number; limit?: number; status?: string }) => Promise<void>;
  fetchProject: (id: string) => Promise<void>;
  createProject: (projectData: any) => Promise<ValuationProject>;
  updateProject: (id: string, projectData: any) => Promise<void>;
  deleteProject: (id: string) => Promise<void>;
  setCurrentProject: (project: ValuationProject | null) => void;
  addScenario: (projectId: string, scenario: ValuationScenario) => void;
  updateScenario: (projectId: string, scenarioId: string, scenario: Partial<ValuationScenario>) => void;
  removeScenario: (projectId: string, scenarioId: string) => void;
  addDocument: (projectId: string, document: Document) => void;
  removeDocument: (projectId: string, documentId: string) => void;
  clearError: () => void;
}

export const useProjectsStore = create<ProjectsState>((set, get) => ({
  projects: [],
  currentProject: null,
  isLoading: false,
  error: null,
  pagination: {
    page: 1,
    limit: 10,
    total: 0,
    totalPages: 0,
  },

  fetchProjects: async (params) => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiClient.getProjects(params);
      
      set({
        projects: response.data,
        pagination: response.pagination,
        isLoading: false,
      });
    } catch (error: any) {
      set({
        error: error.message || 'Failed to fetch projects',
        isLoading: false,
      });
    }
  },

  fetchProject: async (id: string) => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiClient.getProject(id);
      const project = response.data;
      
      set({
        currentProject: project,
        isLoading: false,
      });
      
      // Update project in list if it exists
      const { projects } = get();
      const updatedProjects = projects.map(p => 
        p.id === project.id ? project : p
      );
      set({ projects: updatedProjects });
    } catch (error: any) {
      set({
        error: error.message || 'Failed to fetch project',
        isLoading: false,
      });
    }
  },

  createProject: async (projectData: any) => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiClient.createProject(projectData);
      const newProject = response.data;
      
      const { projects } = get();
      set({
        projects: [newProject, ...projects],
        isLoading: false,
      });
      
      return newProject;
    } catch (error: any) {
      set({
        error: error.message || 'Failed to create project',
        isLoading: false,
      });
      throw error;
    }
  },

  updateProject: async (id: string, projectData: any) => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiClient.updateProject(id, projectData);
      const updatedProject = response.data;
      
      const { projects, currentProject } = get();
      const updatedProjects = projects.map(p => 
        p.id === id ? updatedProject : p
      );
      
      set({
        projects: updatedProjects,
        currentProject: currentProject?.id === id ? updatedProject : currentProject,
        isLoading: false,
      });
    } catch (error: any) {
      set({
        error: error.message || 'Failed to update project',
        isLoading: false,
      });
      throw error;
    }
  },

  deleteProject: async (id: string) => {
    set({ isLoading: true, error: null });
    
    try {
      await apiClient.deleteProject(id);
      
      const { projects, currentProject } = get();
      const filteredProjects = projects.filter(p => p.id !== id);
      
      set({
        projects: filteredProjects,
        currentProject: currentProject?.id === id ? null : currentProject,
        isLoading: false,
      });
    } catch (error: any) {
      set({
        error: error.message || 'Failed to delete project',
        isLoading: false,
      });
      throw error;
    }
  },

  setCurrentProject: (project: ValuationProject | null) => {
    set({ currentProject: project });
  },

  addScenario: (projectId: string, scenario: ValuationScenario) => {
    const { projects, currentProject } = get();
    
    const updateProject = (p: ValuationProject) => {
      if (p.id === projectId) {
        return {
          ...p,
          scenarios: [...p.scenarios, scenario],
        };
      }
      return p;
    };
    
    set({
      projects: projects.map(updateProject),
      currentProject: currentProject ? updateProject(currentProject) : null,
    });
  },

  updateScenario: (projectId: string, scenarioId: string, scenarioUpdate: Partial<ValuationScenario>) => {
    const { projects, currentProject } = get();
    
    const updateProject = (p: ValuationProject) => {
      if (p.id === projectId) {
        return {
          ...p,
          scenarios: p.scenarios.map(s => 
            s.id === scenarioId ? { ...s, ...scenarioUpdate } : s
          ),
        };
      }
      return p;
    };
    
    set({
      projects: projects.map(updateProject),
      currentProject: currentProject ? updateProject(currentProject) : null,
    });
  },

  removeScenario: (projectId: string, scenarioId: string) => {
    const { projects, currentProject } = get();
    
    const updateProject = (p: ValuationProject) => {
      if (p.id === projectId) {
        return {
          ...p,
          scenarios: p.scenarios.filter(s => s.id !== scenarioId),
        };
      }
      return p;
    };
    
    set({
      projects: projects.map(updateProject),
      currentProject: currentProject ? updateProject(currentProject) : null,
    });
  },

  addDocument: (projectId: string, document: Document) => {
    const { projects, currentProject } = get();
    
    const updateProject = (p: ValuationProject) => {
      if (p.id === projectId) {
        return {
          ...p,
          documents: [...p.documents, document],
        };
      }
      return p;
    };
    
    set({
      projects: projects.map(updateProject),
      currentProject: currentProject ? updateProject(currentProject) : null,
    });
  },

  removeDocument: (projectId: string, documentId: string) => {
    const { projects, currentProject } = get();
    
    const updateProject = (p: ValuationProject) => {
      if (p.id === projectId) {
        return {
          ...p,
          documents: p.documents.filter(d => d.id !== documentId),
        };
      }
      return p;
    };
    
    set({
      projects: projects.map(updateProject),
      currentProject: currentProject ? updateProject(currentProject) : null,
    });
  },

  clearError: () => {
    set({ error: null });
  },
}));