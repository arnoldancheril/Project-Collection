import React, { useState } from 'react';
import { View, StyleSheet, Alert } from 'react-native';
import TextField from '../common/TextField';
import ButtonPrimary from '../common/ButtonPrimary';
import ButtonOutlined from '../common/ButtonOutlined';
import { spacing } from '../../styles/theme';

interface LoginFormProps {
  onSubmit?: (data: { email: string; password: string }) => void;
  onGoogleAuth?: () => void;
  loading?: boolean;
}

const LoginForm: React.FC<LoginFormProps> = ({
  onSubmit,
  onGoogleAuth,
  loading = false,
}) => {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  const [errors, setErrors] = useState<{[key: string]: string}>({});

  const validateForm = () => {
    const newErrors: {[key: string]: string} = {};

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!emailRegex.test(formData.email)) {
      newErrors.email = 'Please enter a valid email';
    }

    // Password validation
    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = () => {
    if (!validateForm()) return;
    
    if (onSubmit) {
      onSubmit(formData);
    } else {
      Alert.alert('Success', 'Form validation passed!');
    }
  };

  const handleGoogleAuth = () => {
    if (onGoogleAuth) {
      onGoogleAuth();
    } else {
      Alert.alert('Google Auth', 'Google authentication would be implemented here');
    }
  };

  return (
    <View style={styles.container}>
      <TextField
        label="Email"
        value={formData.email}
        onChangeText={(text) => setFormData({...formData, email: text})}
        error={errors.email}
        keyboardType="email-address"
        autoCapitalize="none"
      />
      
      <TextField
        label="Password"
        value={formData.password}
        onChangeText={(text) => setFormData({...formData, password: text})}
        error={errors.password}
        isPassword={true}
      />

      <ButtonPrimary
        title="Log in"
        onPress={handleSubmit}
        loading={loading}
        disabled={loading}
      />

      <ButtonOutlined
        title="Continue with Google"
        onPress={handleGoogleAuth}
        icon="logo-google"
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
  },
});

export default LoginForm; 