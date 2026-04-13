import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Platform } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { StatusBar } from 'expo-status-bar';

const TestScreen = ({ navigation }) => {
  const [testResults, setTestResults] = useState({});
  
  const runTest = async (testName, testFunction) => {
    try {
      setTestResults(prev => ({
        ...prev,
        [testName]: { status: 'running' }
      }));
      
      const result = await testFunction();
      
      setTestResults(prev => ({
        ...prev,
        [testName]: { status: 'success', result }
      }));
    } catch (error) {
      setTestResults(prev => ({
        ...prev,
        [testName]: { status: 'error', error: error.message }
      }));
    }
  };
  
  const testExpoVersion = async () => {
    return Platform.constants.reactNativeVersion;
  };
  
  const testDevice = async () => {
    return {
      os: Platform.OS,
      version: Platform.Version,
      isDevice: true
    };
  };
  
  const renderTestStatus = (testName) => {
    const test = testResults[testName];
    if (!test) {
      return (
        <Text style={styles.statusPending}>Not Run</Text>
      );
    }
    
    if (test.status === 'running') {
      return (
        <Text style={styles.statusRunning}>Running...</Text>
      );
    }
    
    if (test.status === 'error') {
      return (
        <View style={styles.errorContainer}>
          <Text style={styles.statusError}>Failed</Text>
          <Text style={styles.errorText}>{test.error}</Text>
        </View>
      );
    }
    
    return (
      <View>
        <Text style={styles.statusSuccess}>Success</Text>
        <Text style={styles.resultText}>
          {typeof test.result === 'object' 
            ? JSON.stringify(test.result, null, 2) 
            : test.result}
        </Text>
      </View>
    );
  };
  
  return (
    <View style={styles.container}>
      <StatusBar style="light" />
      
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Ionicons name="arrow-back" size={24} color="#FFFFFF" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Expo Test Screen</Text>
        <View style={{ width: 40 }} />
      </View>
      
      <ScrollView style={styles.content}>
        <Text style={styles.sectionTitle}>Expo SDK Tests</Text>
        
        <View style={styles.testCard}>
          <View style={styles.testHeader}>
            <Text style={styles.testName}>Expo/React Native Version</Text>
            <TouchableOpacity 
              style={styles.runButton}
              onPress={() => runTest('expoVersion', testExpoVersion)}
            >
              <Text style={styles.runButtonText}>Run Test</Text>
            </TouchableOpacity>
          </View>
          
          <View style={styles.resultContainer}>
            {renderTestStatus('expoVersion')}
          </View>
        </View>
        
        <View style={styles.testCard}>
          <View style={styles.testHeader}>
            <Text style={styles.testName}>Device Info</Text>
            <TouchableOpacity 
              style={styles.runButton}
              onPress={() => runTest('deviceInfo', testDevice)}
            >
              <Text style={styles.runButtonText}>Run Test</Text>
            </TouchableOpacity>
          </View>
          
          <View style={styles.resultContainer}>
            {renderTestStatus('deviceInfo')}
          </View>
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#1565C0',
    paddingTop: 50,
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  backButton: {
    width: 40,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  content: {
    flex: 1,
    padding: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
    color: '#212121',
  },
  testCard: {
    backgroundColor: '#FFFFFF',
    padding: 16,
    borderRadius: 8,
    marginBottom: 16,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 1.41,
  },
  testHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  testName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#212121',
  },
  runButton: {
    backgroundColor: '#1565C0',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 4,
  },
  runButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '500',
  },
  resultContainer: {
    borderTopWidth: 1,
    borderTopColor: '#EEEEEE',
    paddingTop: 12,
  },
  statusPending: {
    color: '#9E9E9E',
    fontStyle: 'italic',
  },
  statusRunning: {
    color: '#2196F3',
    fontWeight: '600',
  },
  statusSuccess: {
    color: '#4CAF50',
    fontWeight: '600',
    marginBottom: 8,
  },
  statusError: {
    color: '#F44336',
    fontWeight: '600',
    marginBottom: 4,
  },
  errorContainer: {
    backgroundColor: '#FFEBEE',
    padding: 8,
    borderRadius: 4,
  },
  errorText: {
    color: '#D32F2F',
    fontSize: 14,
  },
  resultText: {
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    fontSize: 14,
    color: '#424242',
    backgroundColor: '#F5F5F5',
    padding: 8,
    borderRadius: 4,
  },
});

export default TestScreen; 