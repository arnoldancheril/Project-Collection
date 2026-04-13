import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, Alert, KeyboardAvoidingView, Platform } from 'react-native';
import { useAuth } from '../../contexts/AuthContext';
import Button from '../../components/Button';
import FormInput from '../../components/FormInput';

const BasicInfoScreen = ({ navigation, route }) => {
  const { profileType } = route.params || {};
  const { updateProfile, userProfile } = useAuth();
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    age: '',
    occupation: '',
    bio: '',
    phone: '',
  });

  const updateFormData = (key, value) => {
    setFormData(prevState => ({
      ...prevState,
      [key]: value
    }));
  };

  const handleContinue = async () => {
    // Validation
    if (!formData.firstName || !formData.lastName || !formData.age) {
      Alert.alert('Error', 'Please fill in all required fields');
      return;
    }

    if (isNaN(parseInt(formData.age)) || parseInt(formData.age) < 18) {
      Alert.alert('Error', 'You must be at least 18 years old');
      return;
    }

    setLoading(true);
    try {
      // Update the profile with basic info
      await updateProfile({
        ...userProfile,
        firstName: formData.firstName,
        lastName: formData.lastName,
        age: parseInt(formData.age),
        occupation: formData.occupation || '',
        bio: formData.bio || '',
        phone: formData.phone || '',
      });

      // Navigate to the next registration step based on profile type
      if (profileType === 'HAVE_ROOM') {
        navigation.navigate('PropertyInfo');
      } else {
        navigation.navigate('LifestyleQuestions');
      }
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to update profile. Please try again.');
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
          <Text style={styles.headerTitle}>Tell us about yourself</Text>
          <Text style={styles.headerSubtitle}>Basic information to set up your profile</Text>
        </View>

        <View style={styles.formContainer}>
          <FormInput
            label="First Name"
            value={formData.firstName}
            onChangeText={(text) => updateFormData('firstName', text)}
            placeholder="Enter your first name"
            required
          />

          <FormInput
            label="Last Name"
            value={formData.lastName}
            onChangeText={(text) => updateFormData('lastName', text)}
            placeholder="Enter your last name"
            required
          />

          <FormInput
            label="Age"
            value={formData.age}
            onChangeText={(text) => updateFormData('age', text)}
            placeholder="Enter your age"
            keyboardType="number-pad"
            required
          />

          <FormInput
            label="Occupation"
            value={formData.occupation}
            onChangeText={(text) => updateFormData('occupation', text)}
            placeholder="What do you do?"
          />

          <FormInput
            label="Phone Number"
            value={formData.phone}
            onChangeText={(text) => updateFormData('phone', text)}
            placeholder="Enter your phone number"
            keyboardType="phone-pad"
          />

          <FormInput
            label="Bio"
            value={formData.bio}
            onChangeText={(text) => updateFormData('bio', text)}
            placeholder="Tell potential roommates about yourself"
            multiline
            numberOfLines={4}
            style={styles.bioInput}
          />

          <Button
            title="Continue"
            onPress={handleContinue}
            loading={loading}
            style={styles.continueButton}
          />
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
  bioInput: {
    height: 100,
    textAlignVertical: 'top',
  },
  continueButton: {
    marginTop: 30,
  },
});

export default BasicInfoScreen; 