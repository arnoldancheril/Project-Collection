import { DefaultTheme } from 'react-native-paper';

export const COLORS = {
  primary: '#4A7AFF',
  lightPrimary: '#E0EBFF',
  secondary: '#FF6B6B',
  background: '#FFFFFF',
  backgroundSecondary: '#F5F8FA',
  text: '#212121',
  textSecondary: '#757575',
  border: '#E0E0E0',
  error: '#F44336',
  success: '#4CAF50',
  warning: '#FFC107',
  white: '#FFFFFF',
  black: '#000000',
  lightGrey: '#CCCCCC',
  grey: '#9E9E9E',
  swipeButtonBg: 'rgba(0, 0, 0, 0.2)',
  filterButton: '#F0F6FF',
  locationPin: '#FF6B6B',
  
  // Gradients
  gradientPrimary: ['#4A7AFF', '#3D6AE8'],
  gradientSecondary: ['#FF6B6B', '#EE5555'],
};

export const FONTS = {
  regular: {
    fontFamily: 'System',
    fontWeight: '400',
  },
  medium: {
    fontFamily: 'System',
    fontWeight: '500',
  },
  semiBold: {
    fontFamily: 'System',
    fontWeight: '600',
  },
  bold: {
    fontFamily: 'System',
    fontWeight: '700',
  }
};

export const SIZES = {
  // Global sizes
  base: 8,
  font: 14,
  radius: 8,
  padding: 24,
  
  // Font sizes
  h1: 30,
  h2: 24,
  h3: 18,
  h4: 16,
  body1: 16,
  body2: 14,
  body3: 12,
  
  // Button sizes
  buttonHeight: 50,
  buttonRadius: 25,
  
  // Card sizes
  cardRadius: 16,
  cardPadding: 16
};

export const SHADOWS = {
  small: {
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  medium: {
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.15,
    shadowRadius: 6,
    elevation: 4,
  },
  large: {
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 6,
    },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 6,
  },
};

export const SPACING = {
  xs: 4,
  s: 8,
  m: 16,
  l: 24,
  xl: 32,
  xxl: 40,
};

const appTheme = { COLORS, FONTS, SIZES, SHADOWS, SPACING };

export default appTheme; 