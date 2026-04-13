import React, { useState } from 'react';
import { View, Text, TextInput, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, SIZES, SHADOWS } from '../utils/theme';

const FormInput = ({
  label,
  placeholder,
  value,
  onChangeText,
  secureTextEntry = false,
  keyboardType = 'default',
  multiline = false,
  numberOfLines = 1,
  error,
  icon,
  containerStyle,
  inputStyle,
  labelStyle,
  required = false,
  editable = true,
  maxLength,
  onBlur,
  onFocus,
  autoCapitalize = 'none',
  rightIcon,
  info,
  testID,
}) => {
  const [isFocused, setIsFocused] = useState(false);
  const [isPasswordVisible, setIsPasswordVisible] = useState(!secureTextEntry);
  
  const handleFocus = () => {
    setIsFocused(true);
    if (onFocus) onFocus();
  };
  
  const handleBlur = () => {
    setIsFocused(false);
    if (onBlur) onBlur();
  };
  
  const togglePasswordVisibility = () => {
    setIsPasswordVisible(!isPasswordVisible);
  };
  
  return (
    <View style={[styles.container, containerStyle]}>
      {label && (
        <View style={styles.labelContainer}>
          <Text style={[styles.label, labelStyle]}>
            {label} {required && <Text style={styles.required}>*</Text>}
          </Text>
          {info && (
            <Text style={styles.infoText}>{info}</Text>
          )}
        </View>
      )}
      
      <View style={[
        styles.inputContainer,
        isFocused ? styles.inputContainerFocused : null,
        error ? styles.inputContainerError : null,
        !editable ? styles.inputContainerDisabled : null,
      ]}>
        {icon && (
          <View style={styles.iconContainer}>
            {icon}
          </View>
        )}
        
        <TextInput
          style={[
            styles.input,
            multiline && styles.multilineInput,
            inputStyle,
          ]}
          placeholder={placeholder}
          value={value}
          onChangeText={onChangeText}
          secureTextEntry={secureTextEntry && !isPasswordVisible}
          keyboardType={keyboardType}
          multiline={multiline}
          numberOfLines={multiline ? numberOfLines : 1}
          editable={editable}
          maxLength={maxLength}
          onFocus={handleFocus}
          onBlur={handleBlur}
          autoCapitalize={autoCapitalize}
          placeholderTextColor={COLORS.placeholder}
          testID={testID}
        />
        
        {secureTextEntry && (
          <TouchableOpacity 
            style={styles.iconContainer} 
            onPress={togglePasswordVisibility}
          >
            <Ionicons 
              name={isPasswordVisible ? 'eye-off-outline' : 'eye-outline'} 
              size={24} 
              color={COLORS.textSecondary} 
            />
          </TouchableOpacity>
        )}
        
        {rightIcon && (
          <View style={styles.iconContainer}>
            {rightIcon}
          </View>
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
    marginBottom: SIZES.padding,
    width: '100%',
  },
  labelContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: SIZES.base / 2,
  },
  label: {
    fontSize: SIZES.font,
    color: COLORS.text,
    fontWeight: '500',
  },
  required: {
    color: COLORS.error,
  },
  infoText: {
    fontSize: SIZES.small,
    color: COLORS.textSecondary,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: COLORS.disabled,
    borderRadius: SIZES.radius,
    backgroundColor: COLORS.surface,
    ...SHADOWS.small,
  },
  inputContainerFocused: {
    borderColor: COLORS.primary,
  },
  inputContainerError: {
    borderColor: COLORS.error,
  },
  inputContainerDisabled: {
    backgroundColor: COLORS.background,
    borderColor: COLORS.disabled,
  },
  iconContainer: {
    paddingHorizontal: SIZES.base,
  },
  input: {
    flex: 1,
    paddingVertical: SIZES.base * 1.5,
    paddingHorizontal: SIZES.base,
    fontSize: SIZES.font,
    color: COLORS.text,
  },
  multilineInput: {
    textAlignVertical: 'top',
    minHeight: 100,
  },
  errorText: {
    color: COLORS.error,
    fontSize: SIZES.small,
    marginTop: SIZES.base / 2,
  },
});

export default FormInput; 