import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import { useAuth } from '../../contexts/AuthContext';
import Button from '../../components/Button';
import FormInput from '../../components/FormInput';

const SignupScreen = ({ navigation }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const { register } = useAuth();

  const handleSignup = async () => {
    // Validation
    if (!email || !password || !confirmPassword) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }

    if (password !== confirmPassword) {
      Alert.alert('Error', 'Passwords do not match');
      return;
    }

    if (password.length < 6) {
      Alert.alert('Error', 'Password must be at least 6 characters');
      return;
    }

    setLoading(true);
    try {
      // Create initial user profile with just email
      const initialProfile = {
        email,
        createdAt: new Date().toISOString(),
        profileType: null, // Will be set during onboarding
        onboardingComplete: false
      };
      
      await register(email, password, initialProfile);
      // After successful registration, navigate to profile type selection
      navigation.navigate('ProfileType');
    } catch (error) {
      Alert.alert('Signup Error', error.message || 'Failed to create account. Please try again.');
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
          <Text style={styles.headerTitle}>Create Account</Text>
          <Text style={styles.headerSubtitle}>Let's help you find your perfect roommate</Text>
        </View>

        <View style={styles.formContainer}>
          <FormInput
            label="Email"
            value={email}
            onChangeText={setEmail}
            placeholder="Enter your email"
            keyboardType="email-address"
            autoCapitalize="none"
          />

          <FormInput
            label="Password"
            value={password}
            onChangeText={setPassword}
            placeholder="Create a password"
            secureTextEntry
          />

          <FormInput
            label="Confirm Password"
            value={confirmPassword}
            onChangeText={setConfirmPassword}
            placeholder="Confirm your password"
            secureTextEntry
          />

          <Button
            title="Sign Up"
            onPress={handleSignup}
            loading={loading}
            style={styles.signupButton}
          />

          <View style={styles.termsContainer}>
            <Text style={styles.termsText}>
              By signing up, you agree to our{' '}
              <Text 
                style={styles.termsLink}
                onPress={() => navigation.navigate('Terms')}
              >
                Terms of Service
              </Text>
              {' '}and{' '}
              <Text 
                style={styles.termsLink}
                onPress={() => navigation.navigate('Privacy')}
              >
                Privacy Policy
              </Text>
            </Text>
          </View>

          <View style={styles.loginContainer}>
            <Text style={styles.loginText}>Already have an account? </Text>
            <TouchableOpacity onPress={() => navigation.navigate('Login')}>
              <Text style={styles.loginLink}>Log In</Text>
            </TouchableOpacity>
          </View>
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
  signupButton: {
    marginTop: 20,
  },
  termsContainer: {
    marginTop: 20,
  },
  termsText: {
    fontSize: 14,
    color: '#757575',
    textAlign: 'center',
  },
  termsLink: {
    color: '#1565C0',
    fontWeight: 'bold',
  },
  loginContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 30,
  },
  loginText: {
    color: '#757575',
    fontSize: 14,
  },
  loginLink: {
    color: '#1565C0',
    fontWeight: 'bold',
    fontSize: 14,
  },
});

export default SignupScreen; 