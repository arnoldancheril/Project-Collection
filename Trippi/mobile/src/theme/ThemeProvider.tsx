/**
 * File: src/theme/ThemeProvider.tsx
 * Purpose: App-wide theming (colors, spacing, typography) via React context and styled primitives.
 */
import React, { createContext, useContext } from 'react';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';

export type Theme = {
  colors: {
    background: string;
    surface: string;
    primary: string;
    textPrimary: string;
    textSecondary: string;
    border: string;
    positive: string;
    warning: string;
  };
  spacing: (multiplier?: number) => number;
  radius: {
    sm: number;
    md: number;
    lg: number;
    xl: number;
  };
};

const defaultTheme: Theme = {
  colors: {
    background: '#0E0F13',
    surface: '#151923',
    primary: '#7C5CFF',
    textPrimary: '#F2F4F7',
    textSecondary: '#98A2B3',
    border: '#1C1F26',
    positive: '#22C55E',
    warning: '#F59E0B',
  },
  spacing: (m: number = 1) => 8 * m,
  radius: { sm: 8, md: 12, lg: 16, xl: 24 },
};

const ThemeContext = createContext<Theme>(defaultTheme);

export function useTheme() {
  return useContext(ThemeContext);
}

type Props = { children: React.ReactNode };

export function ThemeProvider({ children }: Props) {
  return (
    <ThemeContext.Provider value={defaultTheme}>
      <SafeAreaProvider>
        <StatusBar style="light" />
        {children}
      </SafeAreaProvider>
    </ThemeContext.Provider>
  );
}


