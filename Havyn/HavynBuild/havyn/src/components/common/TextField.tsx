import React, { useState } from 'react';
import {
  View,
  TextInput,
  Text,
  TouchableOpacity,
  StyleSheet,
  TextInputProps,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, fontSizes } from '../../styles/theme';

interface TextFieldProps extends TextInputProps {
  label: string;
  error?: string;
  isPassword?: boolean;
  onTogglePassword?: () => void;
}

const TextField: React.FC<TextFieldProps> = ({
  label,
  error,
  isPassword = false,
  value,
  onChangeText,
  ...props
}) => {
  const [showPassword, setShowPassword] = useState(false);
  const [isFocused, setIsFocused] = useState(false);

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  return (
    <View style={styles.container}>
      <View style={[
        styles.inputContainer,
        isFocused && styles.focused,
        error && styles.error
      ]}>
        <TextInput
          style={styles.input}
          value={value}
          onChangeText={onChangeText}
          placeholder={label}
          placeholderTextColor={colors.text.secondary}
          secureTextEntry={isPassword && !showPassword}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          autoCapitalize={isPassword ? 'none' : 'sentences'}
          autoCorrect={false}
          keyboardType={label.toLowerCase().includes('email') ? 'email-address' : 'default'}
          textContentType={label.toLowerCase().includes('email') ? 'emailAddress' : isPassword ? 'password' : 'none'}
          autoComplete={label.toLowerCase().includes('email') ? 'email' : isPassword ? 'password' : 'off'}
          {...props}
        />
        {isPassword && (
          <TouchableOpacity
            style={styles.eyeIcon}
            onPress={togglePasswordVisibility}
            accessibilityLabel={showPassword ? 'Hide password' : 'Show password'}
          >
            <Ionicons
              name={showPassword ? 'eye-off' : 'eye'}
              size={20}
              color={colors.text.secondary}
            />
          </TouchableOpacity>
        )}
      </View>
      {error && (
        <Text style={styles.errorText}>{error}</Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: spacing.md,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: borderRadius.sm,
    paddingHorizontal: spacing.md,
    height: 50,
    backgroundColor: '#FFFFFF',
  },
  focused: {
    borderColor: colors.primary,
    borderWidth: 2,
  },
  error: {
    borderColor: colors.text.error,
  },
  input: {
    flex: 1,
    fontSize: fontSizes.md,
    color: colors.text.primary,
    paddingVertical: spacing.sm,
  },
  eyeIcon: {
    padding: spacing.xs,
  },
  errorText: {
    fontSize: fontSizes.xs,
    color: colors.text.error,
    marginTop: spacing.xs,
    marginLeft: spacing.sm,
  },
});

export default TextField; 