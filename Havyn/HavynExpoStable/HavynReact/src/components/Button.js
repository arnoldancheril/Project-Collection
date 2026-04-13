import React from 'react';
import { StyleSheet, TouchableOpacity, Text, ActivityIndicator, View } from 'react-native';
import { COLORS, SIZES, SHADOWS } from '../utils/theme';

const Button = ({
  title,
  onPress,
  style,
  textStyle,
  disabled = false,
  loading = false,
  icon,
  variant = 'filled', // filled, outlined, text
  size = 'medium', // small, medium, large
  fullWidth = false,
  iconPosition = 'left', // left, right
}) => {
  const getButtonStyles = () => {
    let buttonStyles = [styles.button];
    
    // Add variant styles
    if (variant === 'filled') {
      buttonStyles.push(styles.filledButton);
    } else if (variant === 'outlined') {
      buttonStyles.push(styles.outlinedButton);
    } else if (variant === 'text') {
      buttonStyles.push(styles.textButton);
    }
    
    // Add size styles
    if (size === 'small') {
      buttonStyles.push(styles.smallButton);
    } else if (size === 'large') {
      buttonStyles.push(styles.largeButton);
    }
    
    // Add full width
    if (fullWidth) {
      buttonStyles.push(styles.fullWidth);
    }
    
    // Add disabled style
    if (disabled) {
      buttonStyles.push(styles.disabledButton);
    }
    
    // Add custom styles
    if (style) {
      buttonStyles.push(style);
    }
    
    return buttonStyles;
  };
  
  const getTextStyles = () => {
    let textStyles = [styles.buttonText];
    
    // Add variant text styles
    if (variant === 'outlined') {
      textStyles.push(styles.outlinedButtonText);
    } else if (variant === 'text') {
      textStyles.push(styles.textButtonText);
    }
    
    // Add size text styles
    if (size === 'small') {
      textStyles.push(styles.smallButtonText);
    } else if (size === 'large') {
      textStyles.push(styles.largeButtonText);
    }
    
    // Add disabled text style
    if (disabled) {
      textStyles.push(styles.disabledButtonText);
    }
    
    // Add custom text styles
    if (textStyle) {
      textStyles.push(textStyle);
    }
    
    return textStyles;
  };
  
  return (
    <TouchableOpacity
      style={getButtonStyles()}
      onPress={onPress}
      disabled={disabled || loading}
      activeOpacity={0.7}
    >
      {loading ? (
        <ActivityIndicator size="small" color={variant === 'filled' ? COLORS.surface : COLORS.primary} />
      ) : (
        <View style={styles.contentContainer}>
          {icon && iconPosition === 'left' && <View style={styles.iconLeft}>{icon}</View>}
          <Text style={getTextStyles()}>{title}</Text>
          {icon && iconPosition === 'right' && <View style={styles.iconRight}>{icon}</View>}
        </View>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    borderRadius: SIZES.radius,
    paddingVertical: SIZES.base * 1.5,
    paddingHorizontal: SIZES.padding,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    ...SHADOWS.small,
  },
  filledButton: {
    backgroundColor: COLORS.primary,
  },
  outlinedButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: COLORS.primary,
  },
  textButton: {
    backgroundColor: 'transparent',
    shadowColor: 'transparent',
    elevation: 0,
    paddingHorizontal: SIZES.base,
  },
  smallButton: {
    paddingVertical: SIZES.base,
    paddingHorizontal: SIZES.padding / 1.5,
  },
  largeButton: {
    paddingVertical: SIZES.base * 2,
    paddingHorizontal: SIZES.padding * 1.5,
  },
  fullWidth: {
    width: '100%',
  },
  disabledButton: {
    backgroundColor: COLORS.disabled,
    borderColor: COLORS.disabled,
    opacity: 0.7,
  },
  buttonText: {
    color: COLORS.surface,
    fontSize: SIZES.font,
    fontWeight: '600',
  },
  outlinedButtonText: {
    color: COLORS.primary,
  },
  textButtonText: {
    color: COLORS.primary,
  },
  smallButtonText: {
    fontSize: SIZES.small,
  },
  largeButtonText: {
    fontSize: SIZES.large,
  },
  disabledButtonText: {
    color: COLORS.textSecondary,
  },
  contentContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconLeft: {
    marginRight: SIZES.base,
  },
  iconRight: {
    marginLeft: SIZES.base,
  },
});

export default Button; 