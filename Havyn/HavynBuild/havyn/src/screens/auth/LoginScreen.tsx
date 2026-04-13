import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  ScrollView,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import TextField from '../../components/common/TextField';
import ButtonPrimary from '../../components/common/ButtonPrimary';
import ButtonOutlined from '../../components/common/ButtonOutlined';
import RoommateIllustration from '../../components/common/RoommateIllustration';
import { colors, spacing, borderRadius, fontSizes, fontWeights, shadows } from '../../styles/theme';
import { RootStackParamList } from '../../navigation';

// Define the navigation prop type
type LoginScreenNavigationProp = StackNavigationProp<RootStackParamList>;

const LoginScreen = () => {
  const navigation = useNavigation<LoginScreenNavigationProp>();
  const [isSignUp, setIsSignUp] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  const [errors, setErrors] = useState<{[key: string]: string}>({});
  const [loading, setLoading] = useState(false);

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

    // Sign up specific validations
    if (isSignUp) {
      if (!formData.name) {
        newErrors.name = 'Name is required';
      }
      if (!formData.confirmPassword) {
        newErrors.confirmPassword = 'Please confirm your password';
      } else if (formData.password !== formData.confirmPassword) {
        newErrors.confirmPassword = 'Passwords do not match';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async () => {
    // For testing purposes, navigate directly to Main without validation
    navigation.navigate('Main');
    return;

    /* Original code commented out for testing
    if (!validateForm()) return;

    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      Alert.alert(
        'Success',
        isSignUp ? 'Account created successfully!' : 'Login successful!',
        [{ text: 'OK' }]
      );
    } catch (error) {
      Alert.alert('Error', 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
    */
  };

  const handleGoogleAuth = () => {
    // For testing purposes, navigate directly to Main
    navigation.navigate('Main');
    // Alert.alert('Google Auth', 'Google authentication would be implemented here');
  };

  const toggleAuthMode = () => {
    setIsSignUp(!isSignUp);
    setFormData({ name: '', email: '', password: '', confirmPassword: '' });
    setErrors({});
  };

  return (
    <KeyboardAvoidingView 
      style={styles.container} 
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <LinearGradient
        colors={colors.backgroundGradient as [string, string]}
        style={styles.gradient}
      >
        <SafeAreaView style={styles.safeArea}>
          <ScrollView 
            style={styles.scrollView}
            contentContainerStyle={styles.scrollContent}
            showsVerticalScrollIndicator={false}
          >
            {/* Header */}
            <View style={styles.header}>
              <TouchableOpacity style={styles.logoContainer}>
                <Ionicons name="home" size={48} color={colors.primary} />
              </TouchableOpacity>
            </View>

            {/* Hero Section */}
            <View style={styles.heroSection}>
              <Text style={styles.welcomeText}>
                {isSignUp ? 'Create your Havyn account' : 'Welcome to Havyn'}
              </Text>
              <RoommateIllustration width={240} height={160} />
            </View>

            {/* Auth Form Card */}
            <View style={styles.formCard}>
              {isSignUp && (
                <TextField
                  label="Name"
                  value={formData.name}
                  onChangeText={(text) => setFormData({...formData, name: text})}
                  error={errors.name}
                />
              )}
              
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

              {isSignUp && (
                <>
                  <TextField
                    label="Confirm Password"
                    value={formData.confirmPassword}
                    onChangeText={(text) => setFormData({...formData, confirmPassword: text})}
                    error={errors.confirmPassword}
                    isPassword={true}
                  />
                  
                  <Text style={styles.passwordHint}>
                    Password must be at least 6 characters long
                  </Text>
                </>
              )}

              {/* Primary CTA */}
              <ButtonPrimary
                title={isSignUp ? 'Sign up' : 'Log in'}
                onPress={handleSubmit}
                loading={loading}
                disabled={loading}
              />

              {/* Google Auth */}
              <ButtonOutlined
                title="Continue with Google"
                onPress={handleGoogleAuth}
                icon="logo-google"
              />

              {/* Auth Mode Switcher */}
              <View style={styles.switcherRow}>
                <Text style={styles.switcherText}>
                  {isSignUp ? 'Already have an account? ' : "Don't have an account? "}
                </Text>
                <TouchableOpacity onPress={toggleAuthMode}>
                  <Text style={styles.switcherLink}>
                    {isSignUp ? 'Log in' : 'Sign up'}
                  </Text>
                </TouchableOpacity>
              </View>
            </View>

            {/* Footer */}
            <View style={styles.footer}>
              <Text style={styles.footerText}>
                By continuing, you agree to our Terms & Privacy Policy
              </Text>
              <Text style={styles.versionText}>v1.0.0</Text>
            </View>
          </ScrollView>
        </SafeAreaView>
      </LinearGradient>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  gradient: {
    flex: 1,
  },
  safeArea: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: spacing.lg,
  },
  header: {
    paddingTop: spacing.sm,
    paddingBottom: spacing.md,
  },
  logoContainer: {
    alignSelf: 'flex-start',
  },
  heroSection: {
    alignItems: 'center',
    paddingVertical: spacing.xl,
  },
  welcomeText: {
    fontSize: fontSizes.xxl,
    fontWeight: fontWeights.semiBold as any,
    color: colors.text.primary,
    textAlign: 'center',
    marginBottom: spacing.lg,
    lineHeight: 38,
  },
  formCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: borderRadius.md,
    padding: spacing.lg,
    marginVertical: spacing.md,
    ...shadows.medium,
  },
  passwordHint: {
    fontSize: fontSizes.xs,
    color: colors.text.secondary,
    marginTop: -spacing.sm,
    marginBottom: spacing.md,
    marginLeft: spacing.sm,
  },
  switcherRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: spacing.md,
  },
  switcherText: {
    fontSize: fontSizes.sm,
    color: colors.text.secondary,
  },
  switcherLink: {
    fontSize: fontSizes.sm,
    color: colors.primary,
    fontWeight: fontWeights.semiBold as any,
  },
  footer: {
    alignItems: 'center',
    paddingVertical: spacing.lg,
    marginTop: 'auto',
  },
  footerText: {
    fontSize: fontSizes.xs,
    color: colors.text.secondary,
    textAlign: 'center',
    marginBottom: spacing.xs,
  },
  versionText: {
    fontSize: fontSizes.xs,
    color: colors.text.secondary,
    opacity: 0.6,
  },
});

export default LoginScreen; 