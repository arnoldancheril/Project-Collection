import React, { useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Dimensions, StatusBar, Image, Animated } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useFonts, Poppins_300Light, Poppins_600SemiBold } from '@expo-google-fonts/poppins';
import ChicagoSkyline from '../../components/common/ChicagoSkyline';
import LoadingDots from '../../components/common/LoadingDots';
import { colors, fontSizes, spacing } from '../../styles/theme';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

interface LoadingScreenProps {
  onFinish?: () => void;
}

const LoadingScreen = ({ onFinish }: LoadingScreenProps) => {
  const [fontsLoaded] = useFonts({
    Poppins_300Light,
    Poppins_600SemiBold,
  });

  // Create animated value for logo pulsing
  const logoScale = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    // Start pulse animation
    const pulseAnimation = Animated.loop(
      Animated.sequence([
        Animated.timing(logoScale, {
          toValue: 1.03,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(logoScale, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    );
    
    pulseAnimation.start();

    // Simulate loading time (3 seconds max as per design spec)
    const timer = setTimeout(() => {
      if (onFinish) {
        onFinish();
      }
    }, 3000);

    return () => {
      clearTimeout(timer);
      pulseAnimation.stop();
    };
  }, [onFinish, logoScale]);

  if (!fontsLoaded) {
    return null;
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="transparent" translucent />
      
      {/* Background Gradient */}
      <LinearGradient
        colors={colors.backgroundGradient as [string, string]}
        style={StyleSheet.absoluteFillObject}
        start={{ x: 0, y: 0 }}
        end={{ x: 0, y: 1 }}
      />

      {/* Logo Cluster - centered in screen */}
      <View style={styles.logoCluster}>
        <Animated.View style={{
          transform: [{ scale: logoScale }]
        }}>
          <Image 
            source={require('../../assets/images/Logo.png')} 
            style={styles.logoImage}
            resizeMode="contain"
          />
        </Animated.View>
        
        <Text 
          style={styles.wordmark}
          accessibilityLabel="Havyn"
        >
          Havyn
        </Text>
        
        <Text 
          style={styles.tagline}
          accessibilityLabel="Find your perfect roommate in Chicago"
        >
          Find your perfect roommate{'\n'}in Chicago
        </Text>
      </View>

      {/* Chicago Skyline at bottom */}
      <View style={styles.skylineContainer}>
        <ChicagoSkyline color="#B7C9E5" opacity={0.2} />
      </View>

      {/* Loading Dots */}
      <View style={styles.loaderSection}>
        <LoadingDots 
          color={colors.primary} 
          size={8} 
          spacing={12} 
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#E7F3FF', // Fallback color
    justifyContent: 'center',
    alignItems: 'center',
  },
  logoCluster: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.xl,
    marginBottom: screenHeight * 0.15, // Add space between logo and bottom of screen
  },
  logoImage: {
    width: screenWidth * 0.35, // Responsive sizing
    height: screenWidth * 0.35,
    maxWidth: 160,
    maxHeight: 160,
  },
  wordmark: {
    fontFamily: 'Poppins_600SemiBold',
    fontSize: fontSizes.xxxl, // 48pt
    color: colors.primary,
    marginTop: spacing.lg,
    marginBottom: spacing.md,
    textAlign: 'center',
  },
  tagline: {
    fontFamily: 'Poppins_300Light',
    fontSize: fontSizes.lg, // 20pt
    color: colors.primary,
    textAlign: 'center',
    lineHeight: fontSizes.lg * 1.4, // 140% line height
  },
  skylineContainer: {
    position: 'absolute',
    bottom: 50, // Position above the dots
    width: '100%',
    height: screenHeight * 0.2,
  },
  loaderSection: {
    position: 'absolute',
    bottom: 20,
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    paddingBottom: spacing.xl,
  },
});

export default LoadingScreen; 