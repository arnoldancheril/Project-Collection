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
import { useRouter, Link } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ProfileType } from '../../src/models/User';

export default function SignupScreen() {
  const [step, setStep] = useState<'account' | 'profile'>('account');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [name, setName] = useState('');
  const [profileType, setProfileType] = useState<ProfileType | null>(null);
  const [loading, setLoading] = useState(false);
  
  const router = useRouter();

  const handleAccountSubmit = () => {
    // Validate form
    if (!email || !password || !confirmPassword || !name) {
      Alert.alert('Missing Fields', 'Please fill out all fields');
      return;
    }

    if (password !== confirmPassword) {
      Alert.alert('Password Mismatch', 'Passwords do not match');
      return;
    }

    // Move to profile type selection
    setStep('profile');
  };

  const handleProfileTypeSelect = (type: ProfileType) => {
    setProfileType(type);
  };

  const handleSignup = async () => {
    if (!profileType) {
      Alert.alert('Profile Type Required', 'Please select a profile type');
      return;
    }

    setLoading(true);

    try {
      // In a real app, we would use Firebase authentication and create a user profile
      // For now, we'll just simulate a successful registration
      
      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Navigate to the main app (tabs)
      router.replace('/(tabs)');
    } catch (error) {
      Alert.alert('Signup Error', error instanceof Error ? error.message : 'An unknown error occurred');
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
          <View style={styles.header}>
            {step === 'profile' && (
              <TouchableOpacity 
                style={styles.backButton}
                onPress={() => setStep('account')}
              >
                <Ionicons name="arrow-back" size={24} color="#333" />
                <Text style={styles.backText}>Back</Text>
              </TouchableOpacity>
            )}
            <Text style={styles.headerTitle}>Create Account</Text>
          </View>

          {step === 'account' ? (
            <View style={styles.formContainer}>
              <TextInput
                style={styles.input}
                placeholder="Full Name"
                value={name}
                onChangeText={setName}
                placeholderTextColor="#999"
                autoCapitalize="words"
              />
              
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
              
              <TextInput
                style={styles.input}
                placeholder="Confirm Password"
                secureTextEntry
                value={confirmPassword}
                onChangeText={setConfirmPassword}
                placeholderTextColor="#999"
              />
              
              <TouchableOpacity 
                style={styles.continueButton} 
                onPress={handleAccountSubmit}
              >
                <Text style={styles.continueButtonText}>Continue</Text>
              </TouchableOpacity>
              
              <View style={styles.loginContainer}>
                <Text style={styles.loginText}>Already have an account? </Text>
                <Link href="/(auth)/login" asChild>
                  <TouchableOpacity>
                    <Text style={styles.loginLink}>Log in</Text>
                  </TouchableOpacity>
                </Link>
              </View>
            </View>
          ) : (
            <View style={styles.profileTypeContainer}>
              <Text style={styles.profileTypeTitle}>Select Your Profile Type</Text>
              <Text style={styles.profileTypeSubtitle}>
                Choose the option that best describes your housing situation
              </Text>
              
              <TouchableOpacity 
                style={[
                  styles.profileTypeOption,
                  profileType === 'looking_for_room' && styles.selectedProfileType
                ]}
                onPress={() => handleProfileTypeSelect('looking_for_room')}
              >
                <View style={styles.profileTypeIconContainer}>
                  <Ionicons name="search" size={28} color="#3498db" />
                </View>
                <View style={styles.profileTypeTextContainer}>
                  <Text style={styles.profileTypeOptionTitle}>Looking for a Room</Text>
                  <Text style={styles.profileTypeOptionDesc}>I need a place and a roommate...</Text>
                </View>
                {profileType === 'looking_for_room' && (
                  <View style={styles.checkmarkContainer}>
                    <Ionicons name="checkmark-circle" size={24} color="#3498db" />
                  </View>
                )}
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={[
                  styles.profileTypeOption,
                  profileType === 'have_room' && styles.selectedProfileType
                ]}
                onPress={() => handleProfileTypeSelect('have_room')}
              >
                <View style={styles.profileTypeIconContainer}>
                  <Ionicons name="home" size={28} color="#3498db" />
                </View>
                <View style={styles.profileTypeTextContainer}>
                  <Text style={styles.profileTypeOptionTitle}>Have a Room</Text>
                  <Text style={styles.profileTypeOptionDesc}>I have a place and need a roommate...</Text>
                </View>
                {profileType === 'have_room' && (
                  <View style={styles.checkmarkContainer}>
                    <Ionicons name="checkmark-circle" size={24} color="#3498db" />
                  </View>
                )}
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={[
                  styles.profileTypeOption,
                  profileType === 'apartment_listing' && styles.selectedProfileType
                ]}
                onPress={() => handleProfileTypeSelect('apartment_listing')}
              >
                <View style={styles.profileTypeIconContainer}>
                  <Ionicons name="business" size={28} color="#3498db" />
                </View>
                <View style={styles.profileTypeTextContainer}>
                  <Text style={styles.profileTypeOptionTitle}>Apartment Listing</Text>
                  <Text style={styles.profileTypeOptionDesc}>I'm an apartment company listing...</Text>
                </View>
                {profileType === 'apartment_listing' && (
                  <View style={styles.checkmarkContainer}>
                    <Ionicons name="checkmark-circle" size={24} color="#3498db" />
                  </View>
                )}
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={[
                  styles.continueButton,
                  { marginTop: 30 },
                  !profileType && styles.disabledButton
                ]} 
                onPress={handleSignup}
                disabled={!profileType || loading}
              >
                <Text style={styles.continueButtonText}>
                  {loading ? 'Creating Account...' : 'Continue'}
                </Text>
              </TouchableOpacity>
            </View>
          )}
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
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 30,
    position: 'relative',
  },
  backButton: {
    position: 'absolute',
    left: 0,
    flexDirection: 'row',
    alignItems: 'center',
  },
  backText: {
    fontSize: 16,
    color: '#333',
    marginLeft: 4,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
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
  continueButton: {
    backgroundColor: '#0d2150',
    borderRadius: 8,
    padding: 16,
    alignItems: 'center',
    marginTop: 16,
  },
  continueButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  disabledButton: {
    backgroundColor: '#999',
  },
  loginContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 20,
  },
  loginText: {
    color: '#666',
    fontSize: 16,
  },
  loginLink: {
    color: '#0d2150',
    fontSize: 16,
    fontWeight: 'bold',
  },
  profileTypeContainer: {
    width: '100%',
  },
  profileTypeTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
    marginBottom: 10,
  },
  profileTypeSubtitle: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 30,
  },
  profileTypeOption: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#eee',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  selectedProfileType: {
    borderColor: '#3498db',
    backgroundColor: '#f0f8ff',
  },
  profileTypeIconContainer: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#e6f3ff',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  profileTypeTextContainer: {
    flex: 1,
  },
  profileTypeOptionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  profileTypeOptionDesc: {
    fontSize: 14,
    color: '#666',
  },
  checkmarkContainer: {
    marginLeft: 10,
  },
}); 