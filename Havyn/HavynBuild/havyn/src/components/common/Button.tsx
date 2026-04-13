import React from 'react';
import { TouchableOpacity, Text, StyleSheet, ActivityIndicator, ViewStyle, TextStyle } from 'react-native';
import { colors, spacing, borderRadius } from '../../styles/theme';

interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'outlined' | 'link';
  loading?: boolean;
  disabled?: boolean;
  style?: ViewStyle;
  textStyle?: TextStyle;
}

const Button = ({
  title,
  onPress,
  variant = 'primary',
  loading = false,
  disabled = false,
  style,
  textStyle,
}: ButtonProps) => {
  const getButtonStyles = () => {
    switch (variant) {
      case 'outlined':
        return [
          styles.button,
          styles.outlined,
          disabled && styles.disabledOutlined,
          style,
        ];
      case 'link':
        return [styles.link, disabled && styles.disabledLink, style];
      default:
        return [
          styles.button,
          styles.primary,
          disabled && styles.disabledPrimary,
          style,
        ];
    }
  };

  const getTextStyles = () => {
    switch (variant) {
      case 'outlined':
        return [
          styles.text,
          styles.outlinedText,
          disabled && styles.disabledOutlinedText,
          textStyle,
        ];
      case 'link':
        return [
          styles.text,
          styles.linkText,
          disabled && styles.disabledLinkText,
          textStyle,
        ];
      default:
        return [
          styles.text,
          styles.primaryText,
          disabled && styles.disabledPrimaryText,
          textStyle,
        ];
    }
  };

  return (
    <TouchableOpacity
      style={getButtonStyles()}
      onPress={onPress}
      disabled={disabled || loading}
      activeOpacity={0.8}
    >
      {loading ? (
        <ActivityIndicator
          color={variant === 'primary' ? 'white' : colors.primary}
          size="small"
        />
      ) : (
        <Text style={getTextStyles()}>{title}</Text>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    borderRadius: borderRadius.md,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
    alignItems: 'center',
    justifyContent: 'center',
  },
  primary: {
    backgroundColor: colors.primary,
  },
  outlined: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: colors.primary,
  },
  link: {
    backgroundColor: 'transparent',
    paddingVertical: spacing.xs,
    paddingHorizontal: 0,
  },
  text: {
    fontWeight: '600',
  },
  primaryText: {
    color: 'white',
  },
  outlinedText: {
    color: colors.primary,
  },
  linkText: {
    color: colors.primary,
  },
  disabledPrimary: {
    backgroundColor: colors.primary,
    opacity: 0.5,
  },
  disabledOutlined: {
    borderColor: colors.primary,
    opacity: 0.5,
  },
  disabledLink: {
    opacity: 0.5,
  },
  disabledPrimaryText: {
    color: 'white',
  },
  disabledOutlinedText: {
    color: colors.primary,
  },
  disabledLinkText: {
    color: colors.primary,
  },
});

export default Button; 