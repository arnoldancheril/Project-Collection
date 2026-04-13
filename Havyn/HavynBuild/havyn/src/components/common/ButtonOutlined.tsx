import React from 'react';
import {
  TouchableOpacity,
  Text,
  StyleSheet,
  View,
  ViewStyle,
  TextStyle,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, fontSizes, fontWeights } from '../../styles/theme';

interface ButtonOutlinedProps {
  title: string;
  onPress: () => void;
  disabled?: boolean;
  icon?: keyof typeof Ionicons.glyphMap;
  iconSize?: number;
  style?: ViewStyle;
  textStyle?: TextStyle;
}

const ButtonOutlined: React.FC<ButtonOutlinedProps> = ({
  title,
  onPress,
  disabled = false,
  icon,
  iconSize = 24,
  style,
  textStyle,
}) => {
  return (
    <TouchableOpacity
      style={[styles.container, disabled && styles.disabled, style]}
      onPress={onPress}
      disabled={disabled}
      activeOpacity={0.8}
      accessibilityLabel={title}
      accessibilityRole="button"
    >
      <View style={styles.content}>
        {icon && (
          <Ionicons
            name={icon}
            size={iconSize}
            color={colors.text.primary}
            style={styles.icon}
          />
        )}
        <Text style={[styles.text, textStyle]}>{title}</Text>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: borderRadius.md,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
    marginVertical: spacing.sm,
    backgroundColor: '#FFFFFF',
    minHeight: 50,
  },
  disabled: {
    opacity: 0.5,
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  icon: {
    marginRight: spacing.sm,
  },
  text: {
    color: colors.text.primary,
    fontSize: fontSizes.md,
    fontWeight: fontWeights.regular as any,
    textAlign: 'center',
  },
});

export default ButtonOutlined; 