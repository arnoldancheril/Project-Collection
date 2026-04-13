import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Alert } from 'react-native';
import {
  uploadSampleUsersImagesWithStructure,
  getUserImages,
  getStorageStats,
  clearAllSampleUserImages,
  scanAllStorageUsers
} from '../../src/services/imageService';
import { 
  uploadUsersWithSequentialIdsAndImages, 
  createUsersWithSequentialIds, 
  clearSampleUsers 
} from '../../src/services/sampleDataService';
import { Ionicons } from '@expo/vector-icons';

export default function ProfileScreen() {
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');

  const handleSetupCompleteSystem = async () => {
    setLoading(true);
    setStatus('Setting up complete system with sequential user IDs and structured storage...');
    try {
      const users = await uploadUsersWithSequentialIdsAndImages();
      setStatus(`✅ Success! Created ${users.length} users with sequential IDs (00001, 00002, etc.) and proper image storage.`);
      Alert.alert('Setup Complete', `Created ${users.length} users with sequential IDs and connected image storage!`);
    } catch (error) {
      setStatus(`❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      Alert.alert('Error', 'Failed to set up complete system');
    } finally {
      setLoading(false);
    }
  };

  const handleConnectExistingStorage = async () => {
    setLoading(true);
    setStatus('Connecting existing storage folders to users with sequential IDs...');
    try {
      const users = await createUsersWithSequentialIds();
      setStatus(`✅ Success! Connected ${users.length} users to existing storage with sequential IDs.`);
      Alert.alert('Success', `Connected ${users.length} users to existing storage folders with sequential IDs!`);
    } catch (error) {
      setStatus(`❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      Alert.alert('Error', 'Failed to connect existing storage');
    } finally {
      setLoading(false);
    }
  };

  const handleScanStorage = async () => {
    setLoading(true);
    setStatus('Scanning Firebase Storage for user folders...');
    try {
      const storageUsers = await scanAllStorageUsers();
      let totalProfileImages = 0;
      let totalPropertyImages = 0;
      
      storageUsers.forEach(user => {
        totalProfileImages += user.profileImages.length;
        totalPropertyImages += user.propertyImages.length;
      });
      
      setStatus(`✅ Found ${storageUsers.length} user folders with ${totalProfileImages} profile images and ${totalPropertyImages} property images`);
      Alert.alert('Storage Scan', `Found ${storageUsers.length} users\nProfile images: ${totalProfileImages}\nProperty images: ${totalPropertyImages}`);
    } catch (error) {
      setStatus(`❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      Alert.alert('Error', 'Failed to scan storage');
    } finally {
      setLoading(false);
    }
  };

  const handleClearAll = async () => {
    Alert.alert(
      'Clear All Data',
      'This will remove all users and their images from both database and storage. Continue?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear All',
          style: 'destructive',
          onPress: async () => {
            setLoading(true);
            setStatus('Clearing all user data and images...');
            try {
              await clearSampleUsers(); // This now includes structured image cleanup
              setStatus('✅ Successfully cleared all user data and images');
              Alert.alert('Success', 'All user data and images cleared!');
            } catch (error) {
              setStatus(`❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
              Alert.alert('Error', 'Failed to clear data');
            } finally {
              setLoading(false);
            }
          }
        }
      ]
    );
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Development Tools</Text>
        <Text style={styles.subtitle}>Firebase Database & Storage Management</Text>
      </View>

      <View style={styles.section}>
        <View style={styles.sectionTitleContainer}>
          <Ionicons name="key-outline" size={22} color="#6c5ce7" />
          <Text style={styles.sectionTitle}>Sequential User System</Text>
        </View>
        <Text style={styles.sectionDescription}>
          Create users with sequential IDs (00001, 00002, etc.) connected to Firebase Storage
        </Text>
        
        <TouchableOpacity 
          style={[styles.button, styles.primaryButton]} 
          onPress={handleSetupCompleteSystem}
          disabled={loading}
        >
          <Ionicons name="rocket-outline" size={22} color="white" style={styles.buttonIcon} />
          <View style={styles.buttonTextContainer}>
            <Text style={styles.buttonText}>Setup Complete System</Text>
            <Text style={styles.buttonSubtext}>Create users with sequential IDs + upload images</Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.button, styles.secondaryButton]} 
          onPress={handleConnectExistingStorage}
          disabled={loading}
        >
          <Ionicons name="link-outline" size={22} color="white" style={styles.buttonIcon} />
          <View style={styles.buttonTextContainer}>
            <Text style={styles.buttonText}>Connect Existing Storage</Text>
            <Text style={styles.buttonSubtext}>Link sequential users to existing images</Text>
          </View>
        </TouchableOpacity>
      </View>

      <View style={styles.section}>
        <View style={styles.sectionTitleContainer}>
          <Ionicons name="construct-outline" size={22} color="#3498db" />
          <Text style={styles.sectionTitle}>System Management</Text>
        </View>
        
        <TouchableOpacity 
          style={[styles.button, styles.infoButton]} 
          onPress={handleScanStorage}
          disabled={loading}
        >
          <Ionicons name="search-outline" size={22} color="white" style={styles.buttonIcon} />
          <View style={styles.buttonTextContainer}>
            <Text style={styles.buttonText}>Scan Storage</Text>
            <Text style={styles.buttonSubtext}>Check all user folders and images in Firebase Storage</Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.button, styles.dangerButton]} 
          onPress={handleClearAll}
          disabled={loading}
        >
          <Ionicons name="trash-outline" size={22} color="white" style={styles.buttonIcon} />
          <View style={styles.buttonTextContainer}>
            <Text style={styles.buttonText}>Clear All Data</Text>
            <Text style={styles.buttonSubtext}>Remove all users and their images</Text>
          </View>
        </TouchableOpacity>
      </View>

      {status !== '' && (
        <View style={styles.statusContainer}>
          <Text style={styles.statusText}>{status}</Text>
        </View>
      )}

      {loading && (
        <View style={styles.loadingContainer}>
          <Text style={styles.loadingText}>Processing...</Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    padding: 20,
    paddingTop: 60,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
  },
  section: {
    margin: 16,
    padding: 16,
    backgroundColor: 'white',
    borderRadius: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  sectionTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginLeft: 8,
  },
  sectionDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 16,
    lineHeight: 20,
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 8,
    marginBottom: 12,
  },
  buttonIcon: {
    marginRight: 14,
  },
  buttonTextContainer: {
    flex: 1,
  },
  primaryButton: {
    backgroundColor: '#6c5ce7',
  },
  secondaryButton: {
    backgroundColor: '#2ecc71',
  },
  infoButton: {
    backgroundColor: '#3498db',
  },
  dangerButton: {
    backgroundColor: '#e74c3c',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  buttonSubtext: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 12,
  },
  statusContainer: {
    margin: 16,
    padding: 16,
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#3498db',
  },
  statusText: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
  },
  loadingContainer: {
    margin: 16,
    padding: 20,
    backgroundColor: 'white',
    borderRadius: 8,
    alignItems: 'center',
  },
  loadingText: {
    fontSize: 16,
    color: '#666',
  },
}); 