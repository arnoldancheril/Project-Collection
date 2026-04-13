import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Switch,
  TouchableOpacity,
  Alert,
  SafeAreaView
} from 'react-native';
import { Ionicons, MaterialIcons } from '@expo/vector-icons';
import { useAuth } from '../../contexts/AuthContext';
import { COLORS, SIZES, SHADOWS } from '../../utils/theme';

const SettingItem = ({ title, onPress, icon, IconComponent = Ionicons, value, showArrow = true }) => {
  return (
    <TouchableOpacity style={styles.settingItem} onPress={onPress}>
      <View style={styles.settingIcon}>
        <IconComponent name={icon} size={22} color={COLORS.primary} />
      </View>
      <View style={styles.settingContent}>
        <Text style={styles.settingTitle}>{title}</Text>
        {value && typeof value === 'string' && <Text style={styles.settingValue}>{value}</Text>}
      </View>
      {showArrow && (
        <Ionicons name="chevron-forward" size={20} color={COLORS.textSecondary} />
      )}
      {value && typeof value === 'boolean' && (
        <Switch value={value} />
      )}
    </TouchableOpacity>
  );
};

const SettingsScreen = ({ navigation }) => {
  const { logout } = useAuth();
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [locationEnabled, setLocationEnabled] = useState(true);
  const [autoSchedulingEnabled, setAutoSchedulingEnabled] = useState(false);
  const [instantMessagingEnabled, setInstantMessagingEnabled] = useState(true);
  const [emailNotificationsEnabled, setEmailNotificationsEnabled] = useState(true);
  
  const handleLogout = () => {
    Alert.alert(
      'Logout',
      'Are you sure you want to logout?',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Logout', 
          style: 'destructive', 
          onPress: () => {
            logout();
          } 
        }
      ]
    );
  };
  
  const handleDeleteAccount = () => {
    Alert.alert(
      'Delete Account',
      'Are you sure you want to delete your account? This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Delete', 
          style: 'destructive', 
          onPress: () => {
            // Implement account deletion logic
            Alert.alert('Account Deleted', 'Your account has been successfully deleted.');
            logout();
          } 
        }
      ]
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Settings</Text>
      </View>
      
      <ScrollView style={styles.content}>
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Company Information</Text>
          
          <SettingItem 
            title="Edit Company Profile" 
            icon="business" 
            onPress={() => navigation.navigate('EditProfile')}
          />
          
          <SettingItem 
            title="Contact Person" 
            icon="person-outline" 
            value="John Smith"
            onPress={() => {/* Navigate to edit contact person */}}
          />
          
          <SettingItem 
            title="Business Email" 
            icon="mail-outline" 
            value="contact@company.com"
            onPress={() => {/* Navigate to edit email */}}
          />
          
          <SettingItem 
            title="Business Phone" 
            icon="call-outline" 
            value="(555) 123-4567"
            onPress={() => {/* Navigate to edit phone */}}
          />
          
          <SettingItem 
            title="Business Address" 
            icon="location-outline" 
            value="123 Main St, Suite 400"
            onPress={() => {/* Navigate to edit address */}}
          />
        </View>
        
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Notification Settings</Text>
          
          <SettingItem 
            title="Enable Notifications" 
            icon="notifications-outline" 
            value={notificationsEnabled}
            showArrow={false}
            onPress={() => setNotificationsEnabled(!notificationsEnabled)}
          />
          
          {notificationsEnabled && (
            <>
              <SettingItem 
                title="Instant Messages" 
                icon="chatbubble-outline" 
                value={instantMessagingEnabled}
                showArrow={false}
                onPress={() => setInstantMessagingEnabled(!instantMessagingEnabled)}
              />
              
              <SettingItem 
                title="Email Notifications" 
                icon="mail-outline" 
                value={emailNotificationsEnabled}
                showArrow={false}
                onPress={() => setEmailNotificationsEnabled(!emailNotificationsEnabled)}
              />
              
              <SettingItem 
                title="Auto-Scheduling" 
                icon="calendar-outline" 
                value={autoSchedulingEnabled}
                showArrow={false}
                onPress={() => setAutoSchedulingEnabled(!autoSchedulingEnabled)}
              />
            </>
          )}
        </View>
        
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Listing Settings</Text>
          
          <SettingItem 
            title="Default Listing Duration" 
            icon="time-outline" 
            value="30 days"
            onPress={() => {/* Navigate to listing duration settings */}}
          />
          
          <SettingItem 
            title="Location Services" 
            icon="location-outline" 
            value={locationEnabled}
            showArrow={false}
            onPress={() => setLocationEnabled(!locationEnabled)}
          />
          
          <SettingItem 
            title="Default Currency" 
            icon="cash-outline" 
            value="USD"
            onPress={() => {/* Navigate to currency settings */}}
          />
        </View>
        
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Account Settings</Text>
          
          <SettingItem 
            title="Change Password" 
            icon="key-outline" 
            onPress={() => {/* Navigate to change password */}}
          />
          
          <SettingItem 
            title="Two-Factor Authentication" 
            icon="shield-checkmark-outline" 
            value={false}
            showArrow={false}
            onPress={() => {/* Toggle 2FA */}}
          />
        </View>
        
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Support</Text>
          
          <SettingItem 
            title="Help Center" 
            icon="help-circle-outline" 
            onPress={() => {/* Navigate to help center */}}
          />
          
          <SettingItem 
            title="Contact Support" 
            icon="headset-outline" 
            onPress={() => {/* Navigate to contact form */}}
          />
          
          <SettingItem 
            title="Terms of Service" 
            icon="document-text-outline" 
            onPress={() => {/* Navigate to terms */}}
          />
          
          <SettingItem 
            title="Privacy Policy" 
            icon="lock-closed-outline" 
            onPress={() => {/* Navigate to privacy policy */}}
          />
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Developer</Text>
          
          <SettingItem 
            title="Test Expo Features" 
            icon="build-outline" 
            onPress={() => navigation.navigate('TestExpo')}
          />
        </View>
        
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Account Actions</Text>
          
          <SettingItem 
            title="Logout" 
            icon="log-out-outline" 
            onPress={handleLogout}
            showArrow={false}
          />
          
          <SettingItem 
            title="Delete Account" 
            icon="trash-outline" 
            onPress={handleDeleteAccount}
            showArrow={false}
          />
        </View>
        
        <View style={styles.versionContainer}>
          <Text style={styles.versionText}>Havyn Property Manager v1.0.0</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  header: {
    height: 60,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: COLORS.white,
    ...SHADOWS.medium,
    zIndex: 10,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  headerTitle: {
    fontSize: SIZES.h3,
    fontWeight: 'bold',
    color: COLORS.text,
  },
  content: {
    flex: 1,
  },
  section: {
    marginVertical: 10,
    paddingHorizontal: 15,
    backgroundColor: COLORS.white,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: COLORS.border,
  },
  sectionTitle: {
    fontSize: SIZES.h4,
    fontWeight: 'bold',
    marginVertical: 15,
    color: COLORS.textSecondary,
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
  },
  settingIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: COLORS.lightGray,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 15,
  },
  settingContent: {
    flex: 1,
  },
  settingTitle: {
    fontSize: SIZES.body3,
    color: COLORS.text,
  },
  settingValue: {
    fontSize: SIZES.body4,
    color: COLORS.textSecondary,
    marginTop: 2,
  },
  versionContainer: {
    alignItems: 'center',
    padding: 20,
  },
  versionText: {
    fontSize: SIZES.body4,
    color: COLORS.textSecondary,
  },
});

export default SettingsScreen; 