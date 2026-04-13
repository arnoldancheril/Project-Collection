import React, { useEffect, useRef } from 'react';
import { View, Animated, StyleSheet, Image } from 'react-native';
import { colors } from '../../styles/theme';

interface HavynLogoProps {
  size?: number;
  color?: string;
  showPulse?: boolean;
  showNameLogo?: boolean;
}

const HavynLogo = ({ 
  size = 144, 
  color = colors.primary,
  showPulse = true,
  showNameLogo = false
}: HavynLogoProps) => {
  const scaleValue = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    if (showPulse) {
      const pulseAnimation = Animated.loop(
        Animated.sequence([
          Animated.timing(scaleValue, {
            toValue: 1.03,
            duration: 500,
            useNativeDriver: true,
          }),
          Animated.timing(scaleValue, {
            toValue: 1,
            duration: 1500,
            useNativeDriver: true,
          }),
        ])
      );
      pulseAnimation.start();

      return () => pulseAnimation.stop();
    }
  }, [scaleValue, showPulse]);

  return (
    <Animated.View
      style={[
        styles.container,
        {
          width: showNameLogo ? size * 3 : size,
          height: size,
          transform: [{ scale: scaleValue }],
        },
      ]}
    >
      <Image 
        source={showNameLogo ? require('../../assets/images/Logo_Name.png') : require('../../assets/images/Logo_Icon.png')} 
        style={{
          width: showNameLogo ? size * 3 : size,
          height: size,
          resizeMode: 'contain',
        }}
      />
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
  },
});

export default HavynLogo; 