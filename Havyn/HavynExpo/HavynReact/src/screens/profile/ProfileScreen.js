import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  Image,
  TouchableOpacity,
  ScrollView,
  Switch,
  SafeAreaView,
  Platform
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../../utils/theme';

const ProfileOption = ({ icon, title, subtitle, showArrow = true, onPress }) => (
  <TouchableOpacity style={styles.option} onPress={onPress}>
    <View style={styles.optionIconContainer}>
      <Ionicons name={icon} size={22} color={COLORS.primary} />
    </View>
    <View style={styles.optionContent}>
      <Text style={styles.optionTitle}>{title}</Text>
      {subtitle && <Text style={styles.optionSubtitle}>{subtitle}</Text>}
    </View>
    {showArrow && (
      <Ionicons name="chevron-forward" size={20} color="#C5C5C7" />
    )}
  </TouchableOpacity>
);

const ProfileScreen = () => {
  const [notificationsEnabled, setNotificationsEnabled] = React.useState(true);
  const [darkModeEnabled, setDarkModeEnabled] = React.useState(false);

  const toggleNotifications = () => setNotificationsEnabled(prev => !prev);
  const toggleDarkMode = () => setDarkModeEnabled(prev => !prev);

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Profile</Text>
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Profile Card */}
        <View style={styles.profileCard}>
          <View style={styles.profileHeader}>
            <Image
              source={require('../../../assets/person-placeholder.jpg')}
              style={styles.profileImage}
            />
            <View style={styles.profileInfo}>
              <Text style={styles.profileName}>Your Name</Text>
              <Text style={styles.profileEmail}>email@example.com</Text>
              <TouchableOpacity style={styles.editButton}>
                <Text style={styles.editButtonText}>Edit Profile</Text>
              </TouchableOpacity>
            </View>
          </View>

          <View style={styles.profileCompletion}>
            <View style={styles.completionHeader}>
              <Text style={styles.completionTitle}>Profile Completion</Text>
              <Text style={styles.completionPercentage}>75%</Text>
            </View>
            <View style={styles.progressBar}>
              <View style={[styles.progressFill, { width: '75%' }]} />
            </View>
            <Text style={styles.completionMessage}>
              Add a phone number to complete your profile
            </Text>
          </View>
        </View>

        {/* Account Settings */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Account Settings</Text>
          <View style={styles.sectionContent}>
            <ProfileOption
              icon="person-outline"
              title="Personal Information"
              subtitle="Update your personal details"
              onPress={() => {}}
            />
            <ProfileOption
              icon="home-outline"
              title="Roommate Preferences"
              subtitle="Edit your roommate criteria"
              onPress={() => {}}
            />
            <ProfileOption
              icon="location-outline"
              title="Location Preferences"
              subtitle="Update your preferred areas"
              onPress={() => {}}
            />
          </View>
        </View>

        {/* App Settings */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>App Settings</Text>
          <View style={styles.sectionContent}>
            <TouchableOpacity style={styles.option}>
              <View style={styles.optionIconContainer}>
                <Ionicons name="notifications-outline" size={22} color={COLORS.primary} />
              </View>
              <View style={styles.optionContent}>
                <Text style={styles.optionTitle}>Notifications</Text>
              </View>
              <Switch
                value={notificationsEnabled}
                onValueChange={toggleNotifications}
                trackColor={{ false: '#767577', true: '#c8e6ff' }}
                thumbColor={notificationsEnabled ? COLORS.primary : '#f4f3f4'}
              />
            </TouchableOpacity>
            
            <TouchableOpacity style={styles.option}>
              <View style={styles.optionIconContainer}>
                <Ionicons name="moon-outline" size={22} color={COLORS.primary} />
              </View>
              <View style={styles.optionContent}>
                <Text style={styles.optionTitle}>Dark Mode</Text>
              </View>
              <Switch
                value={darkModeEnabled}
                onValueChange={toggleDarkMode}
                trackColor={{ false: '#767577', true: '#c8e6ff' }}
                thumbColor={darkModeEnabled ? COLORS.primary : '#f4f3f4'}
              />
            </TouchableOpacity>
            
            <ProfileOption
              icon="language-outline"
              title="Language"
              subtitle="English"
              onPress={() => {}}
            />
          </View>
        </View>

        {/* Support & About */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Support & About</Text>
          <View style={styles.sectionContent}>
            <ProfileOption
              icon="help-circle-outline"
              title="Help Center"
              onPress={() => {}}
            />
            <ProfileOption
              icon="document-text-outline"
              title="Terms of Service"
              onPress={() => {}}
            />
            <ProfileOption
              icon="shield-outline"
              title="Privacy Policy"
              onPress={() => {}}
            />
            <ProfileOption
              icon="information-circle-outline"
              title="About Havyn"
              subtitle="Version 1.0.0"
              onPress={() => {}}
            />
          </View>
        </View>

        <TouchableOpacity style={styles.logoutButton}>
          <Text style={styles.logoutText}>Log Out</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f8fa',
  },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 15,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: COLORS.text,
  },
  scrollContent: {
    paddingBottom: 30,
  },
  profileCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    marginHorizontal: 16,
    marginTop: 16,
    padding: 16,
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
      },
      android: {
        elevation: 3,
      },
    }),
  },
  profileHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  profileImage: {
    width: 80,
    height: 80,
    borderRadius: 40,
  },
  profileInfo: {
    marginLeft: 16,
    flex: 1,
  },
  profileName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 4,
  },
  profileEmail: {
    fontSize: 14,
    color: COLORS.textSecondary,
    marginBottom: 8,
  },
  editButton: {
    alignSelf: 'flex-start',
    backgroundColor: COLORS.lightPrimary,
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 20,
  },
  editButtonText: {
    color: COLORS.primary,
    fontSize: 12,
    fontWeight: '600',
  },
  profileCompletion: {
    marginTop: 8,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
  },
  completionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  completionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: COLORS.text,
  },
  completionPercentage: {
    fontSize: 16,
    fontWeight: '600',
    color: COLORS.primary,
  },
  progressBar: {
    height: 8,
    backgroundColor: '#f0f0f0',
    borderRadius: 4,
    marginBottom: 8,
  },
  progressFill: {
    height: '100%',
    backgroundColor: COLORS.primary,
    borderRadius: 4,
  },
  completionMessage: {
    fontSize: 14,
    color: COLORS.textSecondary,
  },
  section: {
    marginTop: 24,
    paddingHorizontal: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
    color: COLORS.text,
  },
  sectionContent: {
    backgroundColor: '#fff',
    borderRadius: 12,
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
      },
      android: {
        elevation: 3,
      },
    }),
  },
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 14,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  optionIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: COLORS.lightPrimary,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  optionContent: {
    flex: 1,
  },
  optionTitle: {
    fontSize: 16,
    color: COLORS.text,
  },
  optionSubtitle: {
    fontSize: 14,
    color: COLORS.textSecondary,
    marginTop: 2,
  },
  logoutButton: {
    marginTop: 30,
    marginHorizontal: 16,
    backgroundColor: '#fff',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#f0f0f0',
  },
  logoutText: {
    fontSize: 16,
    color: '#FF3B30',
    fontWeight: '600',
  },
});

export default ProfileScreen; 