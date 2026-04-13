import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, Alert, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import { useAuth } from '../../contexts/AuthContext';
import Button from '../../components/Button';
import FormInput from '../../components/FormInput';
import { Ionicons } from '@expo/vector-icons';

const LoginScreen = ({ navigation }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }

    setLoading(true);
    try {
      // Special case for admin/password
      if (email === 'admin' && password === 'password') {
        console.log('Admin mode activated');
        Alert.alert('Admin Mode', 'Logging in with special admin privileges');
      }
      
      await login(email, password);
      // On successful login, the AuthContext will update and navigate through AppNavigator
    } catch (error) {
      Alert.alert('Login Error', error.message || 'Failed to login. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const navigateToTest = () => {
    navigation.navigate('Test');
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}
    >
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.logoContainer}>
          <View style={styles.logoCircle}>
            <Ionicons name="home" size={70} color="#1565C0" />
          </View>
          <Text style={styles.appName}>Havyn</Text>
          <Text style={styles.tagline}>Find your perfect roommate match</Text>
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
            placeholder="Enter your password"
            secureTextEntry
          />

          <TouchableOpacity
            onPress={() => navigation.navigate('ForgotPassword')}
            style={styles.forgotPasswordContainer}
          >
            <Text style={styles.forgotPasswordText}>Forgot Password?</Text>
          </TouchableOpacity>

          <Button
            title="Log In"
            onPress={handleLogin}
            loading={loading}
            style={styles.loginButton}
          />

          <TouchableOpacity 
            onPress={() => navigation.navigate('PropertyManagerLogin')} 
            style={styles.propertyManagerButton}
          >
            <Text style={styles.propertyManagerText}>Property Manager Login</Text>
          </TouchableOpacity>

          <View style={styles.signupContainer}>
            <Text style={styles.signupText}>Don't have an account? </Text>
            <TouchableOpacity onPress={() => navigation.navigate('Signup')}>
              <Text style={styles.signupLink}>Sign Up</Text>
            </TouchableOpacity>
          </View>

          {__DEV__ && (
            <TouchableOpacity onPress={navigateToTest} style={styles.testButton}>
              <Text style={styles.testButtonText}>Test Expo Features</Text>
            </TouchableOpacity>
          )}
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
  logoContainer: {
    alignItems: 'center',
    marginBottom: 30,
  },
  logoCircle: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: '#E3F2FD',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 10,
  },
  appName: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#1565C0',
    marginBottom: 8,
  },
  tagline: {
    fontSize: 16,
    color: '#757575',
    textAlign: 'center',
  },
  formContainer: {
    width: '100%',
  },
  forgotPasswordContainer: {
    alignSelf: 'flex-end',
    marginVertical: 10,
  },
  forgotPasswordText: {
    color: '#1565C0',
    fontSize: 14,
  },
  loginButton: {
    marginTop: 10,
  },
  propertyManagerButton: {
    marginTop: 20,
    padding: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#1565C0',
    borderRadius: 8,
  },
  propertyManagerText: {
    color: '#1565C0',
    fontWeight: '600',
    fontSize: 16,
  },
  signupContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 20,
  },
  signupText: {
    color: '#757575',
    fontSize: 14,
  },
  signupLink: {
    color: '#1565C0',
    fontWeight: 'bold',
    fontSize: 14,
  },
  testButton: {
    marginTop: 30,
    padding: 10,
    backgroundColor: '#E0E0E0',
    borderRadius: 5,
    alignItems: 'center',
  },
  testButtonText: {
    color: '#616161',
  },
});

export default LoginScreen; 