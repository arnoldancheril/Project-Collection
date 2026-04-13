import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, ScrollView } from 'react-native';
import { useAuth } from '../../contexts/AuthContext';
import Button from '../../components/Button';
import { Ionicons } from '@expo/vector-icons';

const ProfileTypeScreen = ({ navigation }) => {
  const [selectedType, setSelectedType] = useState(null);
  const [loading, setLoading] = useState(false);
  const { updateProfile, userProfile } = useAuth();

  const profileTypes = [
    {
      id: 'LOOKING_FOR_ROOM',
      title: 'Looking for a Room',
      description: 'Browse available rooms and connect with potential roommates',
      icon: 'search',
    },
    {
      id: 'HAVE_ROOM',
      title: 'I Have a Room',
      description: 'List your room and find the perfect roommate',
      icon: 'home',
    },
    {
      id: 'PROPERTY_MANAGER',
      title: 'Property Manager',
      description: 'List and manage multiple properties',
      icon: 'business',
    },
  ];

  const handleContinue = async () => {
    if (!selectedType) {
      return;
    }

    setLoading(true);
    try {
      await updateProfile({
        ...userProfile,
        profileType: selectedType
      });

      // Navigate to the appropriate next screen based on profile type
      switch (selectedType) {
        case 'LOOKING_FOR_ROOM':
          navigation.navigate('BasicInfo', { profileType: selectedType });
          break;
        case 'HAVE_ROOM':
          navigation.navigate('BasicInfo', { profileType: selectedType });
          break;
        case 'PROPERTY_MANAGER':
          navigation.navigate('ApartmentSignUp');
          break;
        default:
          navigation.navigate('BasicInfo', { profileType: selectedType });
      }
    } catch (error) {
      console.error('Error updating profile type:', error);
    } finally {
      setLoading(false);
    }
  };

  const renderProfileTypeCard = (type) => {
    const isSelected = selectedType === type.id;
    
    return (
      <TouchableOpacity
        key={type.id}
        style={[
          styles.profileTypeCard,
          isSelected && styles.selectedProfileTypeCard
        ]}
        onPress={() => setSelectedType(type.id)}
      >
        <View style={styles.iconContainer}>
          <Ionicons name={type.icon} size={40} color={isSelected ? '#1565C0' : '#757575'} />
        </View>
        <View style={styles.typeTextContainer}>
          <Text style={styles.typeTitle}>{type.title}</Text>
          <Text style={styles.typeDescription}>{type.description}</Text>
        </View>
        <View style={[
          styles.radioButton,
          isSelected && styles.radioButtonSelected
        ]}>
          {isSelected && <View style={styles.radioButtonInner} />}
        </View>
      </TouchableOpacity>
    );
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      <View style={styles.headerContainer}>
        <Text style={styles.headerTitle}>I am...</Text>
        <Text style={styles.headerSubtitle}>Select what best describes you</Text>
      </View>

      <View style={styles.profileTypesContainer}>
        {profileTypes.map(renderProfileTypeCard)}
      </View>

      <Button
        title="Continue"
        onPress={handleContinue}
        disabled={!selectedType}
        loading={loading}
        style={styles.continueButton}
      />
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  contentContainer: {
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
  profileTypesContainer: {
    marginBottom: 20,
  },
  profileTypeCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
    backgroundColor: '#F5F5F5',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  selectedProfileTypeCard: {
    borderColor: '#1565C0',
    backgroundColor: '#E3F2FD',
  },
  iconContainer: {
    width: 60,
    height: 60,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
    borderRadius: 30,
    backgroundColor: '#FFFFFF',
  },
  typeTextContainer: {
    flex: 1,
  },
  typeTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 4,
    color: '#212121',
  },
  typeDescription: {
    fontSize: 14,
    color: '#757575',
  },
  radioButton: {
    width: 24,
    height: 24,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#BDBDBD',
    justifyContent: 'center',
    alignItems: 'center',
  },
  radioButtonSelected: {
    borderColor: '#1565C0',
  },
  radioButtonInner: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#1565C0',
  },
  continueButton: {
    marginTop: 20,
  },
});

export default ProfileTypeScreen; 