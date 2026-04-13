import React from 'react';
import { View, Text, StyleSheet, ActivityIndicator, Image } from 'react-native';
import { COLORS, SIZES } from '../utils/theme';

const LoadingScreen = ({ message = 'Loading...' }) => {
  return (
    <View style={styles.container}>
      <Image
        source={require('../../assets/logo.png')}
        style={styles.logo}
        resizeMode="contain"
      />
      <Text style={styles.appName}>Havyn</Text>
      <ActivityIndicator size="large" color={COLORS.primary} style={styles.spinner} />
      <Text style={styles.loadingText}>{message}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: COLORS.background,
    padding: SIZES.padding,
  },
  logo: {
    width: 120,
    height: 120,
    marginBottom: SIZES.base,
  },
  appName: {
    fontSize: SIZES.xxxlarge,
    fontWeight: 'bold',
    color: COLORS.primary,
    marginBottom: SIZES.padding * 2,
  },
  spinner: {
    marginBottom: SIZES.padding,
  },
  loadingText: {
    fontSize: SIZES.medium,
    color: COLORS.textSecondary,
    textAlign: 'center',
  },
});

export default LoadingScreen; 