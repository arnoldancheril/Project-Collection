import React, { useEffect, useRef } from 'react';
import { View, Animated, StyleSheet, Dimensions } from 'react-native';

const { width: screenWidth } = Dimensions.get('window');

interface ChicagoSkylineProps {
  color?: string;
  opacity?: number;
}

const ChicagoSkyline = ({ 
  color = '#B7C9E5', 
  opacity = 0.2 
}: ChicagoSkylineProps) => {
  const translateX = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const animation = Animated.loop(
      Animated.timing(translateX, {
        toValue: 3,
        duration: 8000,
        useNativeDriver: true,
      })
    );
    animation.start();

    return () => animation.stop();
  }, [translateX]);

  // Simple skyline silhouette using rectangles to mimic Chicago's skyline
  const buildings = [
    { height: 60, width: 20, left: 0 },
    { height: 80, width: 25, left: 25 },
    { height: 120, width: 15, left: 55 }, // Willis Tower (tallest)
    { height: 70, width: 30, left: 75 },
    { height: 90, width: 20, left: 110 },
    { height: 100, width: 25, left: 135 },
    { height: 65, width: 18, left: 165 },
    { height: 85, width: 22, left: 188 },
    { height: 110, width: 20, left: 215 }, // AON Center
    { height: 75, width: 25, left: 240 },
    { height: 95, width: 30, left: 270 },
    { height: 55, width: 20, left: 305 },
    { height: 70, width: 15, left: 330 },
    { height: 80, width: 25, left: 350 },
    { height: 60, width: 20, left: 380 },
  ];

  return (
    <View style={styles.container}>
      <Animated.View
        style={[
          styles.skyline,
          {
            opacity,
            transform: [{ translateX }],
          },
        ]}
      >
        {buildings.map((building, index) => (
          <View
            key={index}
            style={[
              styles.building,
              {
                height: building.height,
                width: building.width,
                left: building.left,
                backgroundColor: color,
              },
            ]}
          />
        ))}
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: 150,
    overflow: 'hidden',
  },
  skyline: {
    position: 'absolute',
    bottom: 0,
    width: screenWidth + 50,
    height: 150,
  },
  building: {
    position: 'absolute',
    bottom: 0,
  },
});

export default ChicagoSkyline; 