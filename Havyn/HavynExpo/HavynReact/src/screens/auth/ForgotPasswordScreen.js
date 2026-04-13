import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import { useAuth } from '../../contexts/AuthContext';
import Button from '../../components/Button';
import FormInput from '../../components/FormInput';

const ForgotPasswordScreen = ({ navigation }) => {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [resetSent, setResetSent] = useState(false);
  const { resetPassword } = useAuth();

  const handleResetPassword = async () => {
    if (!email) {
      Alert.alert('Error', 'Please enter your email address');
      return;
    }

    setLoading(true);
    try {
      await resetPassword(email);
      setResetSent(true);
    } catch (error) {
      Alert.alert('Reset Error', error.message || 'Failed to send reset email. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}
    >
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.headerContainer}>
          <Text style={styles.headerTitle}>Reset Password</Text>
          <Text style={styles.headerSubtitle}>
            {resetSent
              ? "We've sent you an email with password reset instructions"
              : "Enter your email and we'll send you instructions to reset your password"}
          </Text>
        </View>

        {!resetSent ? (
          <View style={styles.formContainer}>
            <FormInput
              label="Email"
              value={email}
              onChangeText={setEmail}
              placeholder="Enter your email address"
              keyboardType="email-address"
              autoCapitalize="none"
            />

            <Button
              title="Send Reset Link"
              onPress={handleResetPassword}
              loading={loading}
              style={styles.resetButton}
            />
          </View>
        ) : (
          <View style={styles.successContainer}>
            <Text style={styles.successText}>
              Please check your email and follow the instructions to reset your password.
            </Text>
            <Button
              title="Return to Login"
              onPress={() => navigation.navigate('Login')}
              style={styles.loginButton}
            />
          </View>
        )}

        <View style={styles.backContainer}>
          <TouchableOpacity onPress={() => navigation.goBack()}>
            <Text style={styles.backLink}>Back to Login</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  scrollContent: {
    flexGrow: 1,
    padding: 20,
    paddingTop: 40,
  },
  headerContainer: {
    marginBottom: 30,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1565C0',
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#757575',
  },
  formContainer: {
    width: '100%',
  },
  resetButton: {
    marginTop: 20,
  },
  successContainer: {
    width: '100%',
    marginTop: 20,
  },
  successText: {
    fontSize: 16,
    color: '#4CAF50',
    textAlign: 'center',
    marginBottom: 20,
  },
  loginButton: {
    marginTop: 20,
  },
  backContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 30,
  },
  backLink: {
    color: '#1565C0',
    fontWeight: 'bold',
    fontSize: 14,
  },
});

export default ForgotPasswordScreen; 