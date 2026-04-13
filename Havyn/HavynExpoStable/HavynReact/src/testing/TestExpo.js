/**
 * TestExpo.js
 * This file provides a simple test screen to verify Expo functionality
 */

import React, { useState, useEffect } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  Button, 
  ScrollView, 
  Platform,
  SafeAreaView,
  StatusBar
} from 'react-native';
import * as FileSystem from 'expo-file-system';
import Constants from 'expo-constants';
import { Camera } from 'expo-camera';
import * as Location from 'expo-location';
import * as Notifications from 'expo-notifications';

const TestExpo = () => {
  const [tests, setTests] = useState({
    platform: { result: Platform.OS, passed: true },
    expoSDK: { result: Constants.expoVersion, passed: !!Constants.expoVersion },
    filesystem: { result: 'Pending...', passed: null },
    camera: { result: 'Pending...', passed: null },
    location: { result: 'Pending...', passed: null },
    notifications: { result: 'Pending...', passed: null }
  });

  useEffect(() => {
    const runTests = async () => {
      // Test FileSystem
      try {
        const docDir = FileSystem.documentDirectory;
        setTests(prev => ({
          ...prev,
          filesystem: { result: `Available: ${docDir}`, passed: true }
        }));
      } catch (error) {
        setTests(prev => ({
          ...prev,
          filesystem: { result: `Error: ${error.message}`, passed: false }
        }));
      }

      // Test Camera
      try {
        const { status } = await Camera.requestCameraPermissionsAsync();
        setTests(prev => ({
          ...prev,
          camera: { 
            result: `Permission: ${status}`, 
            passed: status === 'granted' 
          }
        }));
      } catch (error) {
        setTests(prev => ({
          ...prev,
          camera: { result: `Error: ${error.message}`, passed: false }
        }));
      }

      // Test Location
      try {
        const { status } = await Location.requestForegroundPermissionsAsync();
        setTests(prev => ({
          ...prev,
          location: { 
            result: `Permission: ${status}`, 
            passed: status === 'granted' 
          }
        }));
      } catch (error) {
        setTests(prev => ({
          ...prev,
          location: { result: `Error: ${error.message}`, passed: false }
        }));
      }
      
      // Test Notifications
      try {
        const { status } = await Notifications.requestPermissionsAsync();
        setTests(prev => ({
          ...prev,
          notifications: { 
            result: `Permission: ${status}`, 
            passed: status === 'granted' 
          }
        }));
      } catch (error) {
        setTests(prev => ({
          ...prev,
          notifications: { result: `Error: ${error.message}`, passed: false }
        }));
      }
    };
    
    runTests();
  }, []);

  const renderTestResults = () => {
    return Object.entries(tests).map(([testName, { result, passed }]) => {
      let statusColor = '#888'; // gray for pending
      if (passed === true) statusColor = '#4CAF50'; // green for passed
      if (passed === false) statusColor = '#F44336'; // red for failed
      
      return (
        <View key={testName} style={styles.testRow}>
          <Text style={styles.testName}>{testName}</Text>
          <Text style={[styles.testResult, { color: statusColor }]}>{result}</Text>
        </View>
      );
    });
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <Text style={styles.title}>Expo Functionality Test</Text>
      
      <ScrollView style={styles.scrollView}>
        <View style={styles.testContainer}>
          {renderTestResults()}
        </View>
      </ScrollView>
      
      <View style={styles.footer}>
        <Text style={styles.footerText}>
          Havyn App - Expo SDK {Constants.expoVersion}
        </Text>
        <Text style={styles.instructions}>
          If all tests pass, your environment is properly configured for Expo development.
          {'\n\n'}
          Any failed tests may require installing additional dependencies or configuring permissions.
        </Text>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    textAlign: 'center',
    marginVertical: 20,
    color: '#333',
  },
  scrollView: {
    flex: 1,
    padding: 15,
  },
  testContainer: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  testRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  testName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    textTransform: 'capitalize',
  },
  testResult: {
    fontSize: 14,
    maxWidth: '60%',
  },
  footer: {
    padding: 20,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  footerText: {
    textAlign: 'center',
    color: '#666',
    marginBottom: 10,
  },
  instructions: {
    fontSize: 14,
    color: '#555',
    textAlign: 'center',
    lineHeight: 20,
  },
});

export default TestExpo; 