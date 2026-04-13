// Theme constants based on the design specs

export const colors = {
  primary: '#083C6C', // Primary Navy
  primaryProfile: '#033E6B', // Primary navy for profile screen
  primaryGradient: ['#08457A', '#072F50'], // Gradient for buttons
  accent: '#77B6FF', // Accent lighter blue for icons/indicators
  backgroundGradient: ['#E7F3FF', '#FFFFFF'], // Sky Gradient
  profileBackgroundGradient: ['#E4F2FF', '#FFFFFF'], // Profile screen background gradient
  skylineTint: 'rgba(183, 201, 229, 0.2)', // Skyline Tint with 20% opacity
  text: {
    primary: '#283A46', // Body text
    secondary: '#666666', // Secondary text
    error: '#D64545', // Error text
  },
  border: '#CBD7E6', // Field border
  white: '#FFFFFF',
  cardShadow: 'rgba(3, 62, 107, 0.08)', // 8% opacity shadow for cards
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

export const borderRadius = {
  sm: 4,
  md: 12,
  lg: 24,
  xl: 32,
};

export const fontSizes = {
  xs: 12,
  sm: 14,
  md: 16,
  lg: 20,
  xl: 24,
  xxl: 32,
  xxxl: 48,
};

export const fontWeights = {
  light: '300',
  regular: '400',
  semiBold: '600',
  bold: '700',
};

// Shadow styles
export const shadows = {
  small: {
    shadowColor: colors.primary,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  medium: {
    shadowColor: colors.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 4,
  },
}; 