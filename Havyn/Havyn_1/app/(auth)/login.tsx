import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  Image,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Alert
} from 'react-native';
import { Link, useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';

export default function LoginScreen() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert('Missing Fields', 'Please enter both email and password');
      return;
    }

    setLoading(true);

    try {
      // In a real app, we would use Firebase authentication
      // For now, we'll just simulate a successful login
      
      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Navigate to the main app (tabs)
      router.replace('/(tabs)');
    } catch (error) {
      Alert.alert('Login Error', error instanceof Error ? error.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    setLoading(true);

    try {
      // In a real app, we would use Firebase Google authentication
      // For now, we'll just simulate a successful login
      
      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Navigate to the main app (tabs)
      router.replace('/(tabs)');
    } catch (error) {
      Alert.alert('Google Login Error', error instanceof Error ? error.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <KeyboardAvoidingView 
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.container}
      >
        <ScrollView 
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
        >
          {/* Logo and Header */}
          <View style={styles.logoContainer}>
            <Image 
              source={require('../../assets/images/Logo_Just_Icon.png')}
              style={styles.logo}
              resizeMode="contain"
            />
            <Text style={styles.title}>Welcome to</Text>
            <Text style={styles.appName}>Havyn</Text>
            
            {/* People illustration */}
            <View style={styles.illustrationContainer}>
              <Image 
                source={{ uri: 'https://via.placeholder.com/300x150/e0f2ff/0066cc?text=People+Illustration' }}
                style={styles.illustration}
                resizeMode="contain"
              />
            </View>
          </View>

          {/* Login Form */}
          <View style={styles.formContainer}>
            <TextInput
              style={styles.input}
              placeholder="Email"
              keyboardType="email-address"
              autoCapitalize="none"
              value={email}
              onChangeText={setEmail}
              placeholderTextColor="#999"
            />
            
            <TextInput
              style={styles.input}
              placeholder="Password"
              secureTextEntry
              value={password}
              onChangeText={setPassword}
              placeholderTextColor="#999"
            />
            
            <TouchableOpacity 
              style={styles.loginButton} 
              onPress={handleLogin}
              disabled={loading}
            >
              <Text style={styles.loginButtonText}>
                {loading ? 'Logging in...' : 'Log in'}
              </Text>
            </TouchableOpacity>
            
            <TouchableOpacity 
              style={styles.googleButton}
              onPress={handleGoogleLogin}
              disabled={loading}
            >
              <Image 
                source={{ uri: 'https://developers.google.com/identity/images/g-logo.png' }}
                style={styles.googleIcon}
              />
              <Text style={styles.googleButtonText}>Continue with Google</Text>
            </TouchableOpacity>
            
            <View style={styles.signupContainer}>
              <Text style={styles.signupText}>Don't have an account? </Text>
              <Link href="/(auth)/signup" asChild>
                <TouchableOpacity>
                  <Text style={styles.signupLink}>Sign up</Text>
                </TouchableOpacity>
              </Link>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  container: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    padding: 20,
    justifyContent: 'center',
  },
  logoContainer: {
    alignItems: 'center',
    marginBottom: 30,
  },
  logo: {
    width: 80,
    height: 80,
    marginBottom: 10,
  },
  title: {
    fontSize: 24,
    color: '#333',
    fontWeight: '500',
  },
  appName: {
    fontSize: 32,
    color: '#0d2150',
    fontWeight: 'bold',
  },
  illustrationContainer: {
    marginVertical: 20,
    alignItems: 'center',
  },
  illustration: {
    width: 200,
    height: 120,
  },
  formContainer: {
    width: '100%',
  },
  input: {
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 15,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#ddd',
    fontSize: 16,
  },
  loginButton: {
    backgroundColor: '#0d2150',
    borderRadius: 8,
    padding: 16,
    alignItems: 'center',
    marginBottom: 16,
  },
  loginButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  googleButton: {
    flexDirection: 'row',
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 16,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#ddd',
    marginBottom: 20,
  },
  googleIcon: {
    width: 20,
    height: 20,
    marginRight: 10,
  },
  googleButtonText: {
    color: '#333',
    fontSize: 16,
    fontWeight: '500',
  },
  signupContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 10,
  },
  signupText: {
    color: '#666',
    fontSize: 16,
  },
  signupLink: {
    color: '#0d2150',
    fontSize: 16,
    fontWeight: 'bold',
  },
}); 