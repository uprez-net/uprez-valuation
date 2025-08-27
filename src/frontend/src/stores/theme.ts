import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Theme } from '@/types';

interface ThemeState {
  theme: 'light' | 'dark';
  primaryColor: string;
  secondaryColor: string;
  branding: {
    logo?: string;
    companyName: string;
  };
  
  // Actions
  setTheme: (theme: 'light' | 'dark') => void;
  setPrimaryColor: (color: string) => void;
  setSecondaryColor: (color: string) => void;
  setBranding: (branding: Partial<ThemeState['branding']>) => void;
  applyOrganizationTheme: (orgBranding: any) => void;
  resetTheme: () => void;
}

const defaultBranding = {
  companyName: 'Uprez Valuation',
  logo: undefined,
};

export const useThemeStore = create<ThemeState>()(
  persist(
    (set, get) => ({
      theme: 'light',
      primaryColor: '#0ea5e9',
      secondaryColor: '#0284c7',
      branding: defaultBranding,

      setTheme: (theme: 'light' | 'dark') => {
        set({ theme });
        
        // Apply theme to document
        const root = document.documentElement;
        root.classList.remove('light', 'dark');
        root.classList.add(theme);
      },

      setPrimaryColor: (color: string) => {
        set({ primaryColor: color });
        
        // Apply color to CSS custom properties
        const root = document.documentElement;
        const hsl = hexToHsl(color);
        root.style.setProperty('--primary', `${hsl.h} ${hsl.s}% ${hsl.l}%`);
      },

      setSecondaryColor: (color: string) => {
        set({ secondaryColor: color });
        
        // Apply color to CSS custom properties
        const root = document.documentElement;
        const hsl = hexToHsl(color);
        root.style.setProperty('--secondary', `${hsl.h} ${hsl.s}% ${hsl.l}%`);
      },

      setBranding: (branding: Partial<ThemeState['branding']>) => {
        set((state) => ({
          branding: { ...state.branding, ...branding },
        }));
      },

      applyOrganizationTheme: (orgBranding: any) => {
        if (!orgBranding) return;
        
        const { primaryColor, secondaryColor, logo } = orgBranding;
        
        if (primaryColor) {
          get().setPrimaryColor(primaryColor);
        }
        
        if (secondaryColor) {
          get().setSecondaryColor(secondaryColor);
        }
        
        if (logo) {
          get().setBranding({ logo });
        }
      },

      resetTheme: () => {
        set({
          theme: 'light',
          primaryColor: '#0ea5e9',
          secondaryColor: '#0284c7',
          branding: defaultBranding,
        });
        
        // Reset document classes and styles
        const root = document.documentElement;
        root.classList.remove('dark');
        root.classList.add('light');
        root.style.removeProperty('--primary');
        root.style.removeProperty('--secondary');
      },
    }),
    {
      name: 'theme-storage',
      onRehydrateStorage: () => (state) => {
        if (state) {
          // Apply theme to document after hydration
          const root = document.documentElement;
          root.classList.remove('light', 'dark');
          root.classList.add(state.theme);
          
          // Apply colors
          const primaryHsl = hexToHsl(state.primaryColor);
          const secondaryHsl = hexToHsl(state.secondaryColor);
          
          root.style.setProperty('--primary', `${primaryHsl.h} ${primaryHsl.s}% ${primaryHsl.l}%`);
          root.style.setProperty('--secondary', `${secondaryHsl.h} ${secondaryHsl.s}% ${secondaryHsl.l}%`);
        }
      },
    }
  )
);

// Utility function to convert hex to HSL
function hexToHsl(hex: string): { h: number; s: number; l: number } {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;

  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  
  let h: number, s: number;
  const l = (max + min) / 2;

  if (max === min) {
    h = s = 0; // achromatic
  } else {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    
    switch (max) {
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      case b: h = (r - g) / d + 4; break;
      default: h = 0;
    }
    h /= 6;
  }

  return {
    h: Math.round(h * 360),
    s: Math.round(s * 100),
    l: Math.round(l * 100),
  };
}