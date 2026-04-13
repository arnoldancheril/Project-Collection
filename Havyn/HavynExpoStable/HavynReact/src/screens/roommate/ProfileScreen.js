import React from 'react';
import { View, Text, StyleSheet, ScrollView, Image, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../../utils/theme';

const ProfileScreen = ({ navigation }) => {
  // Mock user data - would come from context/state in a real app
  const user = {
    name: 'Alex Johnson',
    email: 'alex.johnson@example.com',
    bio: 'UX Designer with a passion for minimalist design. Looking for a clean, quiet space near downtown. I enjoy cooking, hiking, and playing guitar in my free time.',
    age: 28,
    gender: 'Male',
    occupation: 'UX Designer at TechCorp',
    preferences: {
      cleanliness: 4,
      noise: 3,
      guests: 2,
      pets: true,
      smoking: false,
    },
    location: 'Chicago, IL',
    moveInDate: 'Flexible, from June 1st',
    budget: '$1200 - $1800 per month',
    profileImage: require('../../../assets/person-placeholder.jpg'),
  };

  const renderPreferenceItem = (icon, label, value) => (
    <View style={styles.preferenceItem}>
      <Ionicons name={icon} size={20} color={COLORS.primary} />
      <View style={styles.preferenceDetail}>
        <Text style={styles.preferenceLabel}>{label}</Text>
        <Text style={styles.preferenceValue}>{value}</Text>
      </View>
    </View>
  );

  const renderPreferenceScale = (label, value, icon) => (
    <View style={styles.preferenceScale}>
      <View style={styles.scaleHeader}>
        <Text style={styles.preferenceLabel}>{label}</Text>
        <Ionicons name={icon} size={18} color={COLORS.primary} />
      </View>
      <View style={styles.scaleBar}>
        {[1, 2, 3, 4, 5].map((item) => (
          <View 
            key={item} 
            style={[
              styles.scaleItem, 
              item <= value ? styles.scaleItemActive : styles.scaleItemInactive
            ]} 
          />
        ))}
      </View>
    </View>
  );

  return (
    <ScrollView style={styles.container}>
      {/* Profile Header */}
      <View style={styles.header}>
        <View style={styles.profileImageContainer}>
          <Image source={user.profileImage} style={styles.profileImage} />
          <TouchableOpacity style={styles.editImageButton}>
            <Ionicons name="camera" size={20} color="#fff" />
          </TouchableOpacity>
        </View>
        
        <Text style={styles.userName}>{user.name}</Text>
        <Text style={styles.userInfo}>{user.age} • {user.gender} • {user.occupation}</Text>
        
        <TouchableOpacity 
          style={styles.editProfileButton}
          onPress={() => navigation.navigate('Settings')}
        >
          <Text style={styles.editProfileText}>Edit Profile</Text>
        </TouchableOpacity>
      </View>
      
      {/* About Section */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>About Me</Text>
          <TouchableOpacity>
            <Ionicons name="pencil" size={20} color={COLORS.primary} />
          </TouchableOpacity>
        </View>
        <Text style={styles.bioText}>{user.bio}</Text>
      </View>
      
      {/* Housing Preferences */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Housing Preferences</Text>
          <TouchableOpacity>
            <Ionicons name="pencil" size={20} color={COLORS.primary} />
          </TouchableOpacity>
        </View>
        
        {renderPreferenceItem('location', 'Location', user.location)}
        {renderPreferenceItem('calendar', 'Move-in Date', user.moveInDate)}
        {renderPreferenceItem('cash-outline', 'Budget', user.budget)}
      </View>
      
      {/* Lifestyle Preferences */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Lifestyle Preferences</Text>
          <TouchableOpacity>
            <Ionicons name="pencil" size={20} color={COLORS.primary} />
          </TouchableOpacity>
        </View>
        
        {renderPreferenceScale('Cleanliness', user.preferences.cleanliness, 'sparkles-outline')}
        {renderPreferenceScale('Noise Level', user.preferences.noise, 'volume-high-outline')}
        {renderPreferenceScale('Guest Frequency', user.preferences.guests, 'people-outline')}
        
        <View style={styles.togglePreferences}>
          <View style={styles.toggleItem}>
            <Ionicons 
              name={user.preferences.pets ? 'checkmark-circle' : 'close-circle'} 
              size={22} 
              color={user.preferences.pets ? COLORS.primary : '#666'} 
            />
            <Text style={styles.toggleLabel}>Pet Friendly</Text>
          </View>
          
          <View style={styles.toggleItem}>
            <Ionicons 
              name={user.preferences.smoking ? 'checkmark-circle' : 'close-circle'} 
              size={22} 
              color={user.preferences.smoking ? COLORS.primary : '#666'} 
            />
            <Text style={styles.toggleLabel}>Smoking Allowed</Text>
          </View>
        </View>
      </View>
      
      {/* Account Options */}
      <View style={styles.accountOptions}>
        <TouchableOpacity style={styles.accountOption}>
          <Ionicons name="notifications-outline" size={24} color="#555" />
          <Text style={styles.accountOptionText}>Notification Settings</Text>
          <Ionicons name="chevron-forward" size={20} color="#ccc" />
        </TouchableOpacity>
        
        <TouchableOpacity style={styles.accountOption}>
          <Ionicons name="lock-closed-outline" size={24} color="#555" />
          <Text style={styles.accountOptionText}>Privacy Settings</Text>
          <Ionicons name="chevron-forward" size={20} color="#ccc" />
        </TouchableOpacity>
        
        <TouchableOpacity style={styles.accountOption}>
          <Ionicons name="help-circle-outline" size={24} color="#555" />
          <Text style={styles.accountOptionText}>Help & Support</Text>
          <Ionicons name="chevron-forward" size={20} color="#ccc" />
        </TouchableOpacity>
        
        <TouchableOpacity style={styles.accountOption}>
          <Ionicons name="log-out-outline" size={24} color="#d9534f" />
          <Text style={[styles.accountOptionText, { color: '#d9534f' }]}>Log Out</Text>
          <Ionicons name="chevron-forward" size={20} color="#ccc" />
        </TouchableOpacity>
      </View>
      
      <View style={styles.footer}>
        <Text style={styles.version}>Version 1.0.0</Text>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f8f8',
  },
  header: {
    alignItems: 'center',
    paddingVertical: 25,
    paddingHorizontal: 20,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  profileImageContainer: {
    position: 'relative',
    marginBottom: 15,
  },
  profileImage: {
    width: 120,
    height: 120,
    borderRadius: 60,
  },
  editImageButton: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    backgroundColor: COLORS.primary,
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#fff',
  },
  userName: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  userInfo: {
    fontSize: 14,
    color: '#666',
    marginBottom: 15,
  },
  editProfileButton: {
    paddingVertical: 8,
    paddingHorizontal: 24,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: COLORS.primary,
  },
  editProfileText: {
    color: COLORS.primary,
    fontWeight: '500',
  },
  section: {
    backgroundColor: '#fff',
    marginTop: 15,
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  bioText: {
    fontSize: 14,
    lineHeight: 22,
    color: '#444',
  },
  preferenceItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 8,
  },
  preferenceDetail: {
    marginLeft: 10,
  },
  preferenceLabel: {
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 2,
  },
  preferenceValue: {
    fontSize: 14,
    color: '#666',
  },
  preferenceScale: {
    marginVertical: 12,
  },
  scaleHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  scaleBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  scaleItem: {
    width: '18%',
    height: 8,
    borderRadius: 4,
  },
  scaleItemActive: {
    backgroundColor: COLORS.primary,
  },
  scaleItemInactive: {
    backgroundColor: '#e0e0e0',
  },
  togglePreferences: {
    marginTop: 15,
  },
  toggleItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 8,
  },
  toggleLabel: {
    marginLeft: 10,
    fontSize: 14,
    color: '#444',
  },
  accountOptions: {
    backgroundColor: '#fff',
    marginTop: 15,
    marginBottom: 20,
  },
  accountOption: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  accountOptionText: {
    flex: 1,
    fontSize: 16,
    marginLeft: 15,
    color: '#333',
  },
  footer: {
    alignItems: 'center',
    paddingBottom: 30,
  },
  version: {
    fontSize: 12,
    color: '#999',
  },
});

export default ProfileScreen; 