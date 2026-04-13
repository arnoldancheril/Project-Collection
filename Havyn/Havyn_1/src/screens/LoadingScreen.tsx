import React from 'react';
import { View, Text, StyleSheet, ActivityIndicator, Dimensions } from 'react-native';
import { Logo } from '../components';
import { colors } from '../constants';

const { width } = Dimensions.get('window');

const LoadingScreen = () => {
  return (
    <View style={styles.container}>
      <View style={styles.logoContainer}>
        <Logo type="horizontal" size={80} />
        <Text style={styles.tagline}>Find your perfect roommate</Text>
        <Text style={styles.location}>in Chicago</Text>
      </View>
      
      <View style={styles.bottomSection}>
        <View style={styles.dotsContainer}>
          <View style={[styles.dot, styles.activeDot]} />
          <View style={styles.dot} />
          <View style={styles.dot} />
        </View>
        
        <ActivityIndicator 
          size="small" 
          color={colors.primary} 
          style={styles.loader} 
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 60,
  },
  logoContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 80,
  },
  tagline: {
    fontSize: 24,
    color: colors.primary,
    fontWeight: '500',
    textAlign: 'center',
    marginTop: 20,
  },
  location: {
    fontSize: 24,
    color: colors.primary,
    fontWeight: '500',
    textAlign: 'center',
    marginTop: 5,
  },
  bottomSection: {
    alignItems: 'center',
  },
  dotsContainer: {
    flexDirection: 'row',
    marginBottom: 30,
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: colors.lightGray,
    marginHorizontal: 5,
  },
  activeDot: {
    backgroundColor: colors.primary,
  },
  loader: {
    marginBottom: 20,
  },
});

export default LoadingScreen; 