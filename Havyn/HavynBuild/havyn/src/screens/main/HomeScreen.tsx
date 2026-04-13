import React, { useRef, useState } from 'react';
import { 
  View, 
  StyleSheet, 
  Dimensions, 
  StatusBar,
  Animated,
  PanResponder,
  Alert 
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';

import HomeHeader from '../../components/common/HomeHeader';
import RoommateCard, { RoommateProfile } from '../../components/main/RoommateCard';
import SwipeHints from '../../components/main/SwipeHints';
import { colors, spacing } from '../../styles/theme';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

// Sample data for testing
const sampleProfiles: RoommateProfile[] = [
  {
    id: '1',
    name: 'Alex',
    age: 34,
    location: 'Lincoln Park',
    price: 900,
    bedrooms: '1 bd',
    gender: 'Male',
    photo: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=400&h=600&fit=crop&crop=face',
    bio: 'Looking for a clean and respectful roommate! I enjoy cooking and weekend farmers markets. Non-smoker and tech professional.',
    isPetFriendly: false,
  },
  {
    id: '2',
    name: 'Lucy',
    age: 26,
    location: 'Wicker Park',
    price: 850,
    bedrooms: '2 bd',
    gender: 'Female',
    photo: 'https://images.unsplash.com/photo-1494790108755-2616b612e8cf?w=400&h=600&fit=crop&crop=face',
    bio: 'Musician and coffee enthusiast. Love to explore the city and try new restaurants. Clean, quiet, and respectful.',
    isPetFriendly: true,
  },
  {
    id: '3',
    name: 'Sarah',
    age: 28,
    location: 'Logan Square',
    price: 1100,
    bedrooms: '1 bd',
    gender: 'Female',
    photo: 'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=400&h=600&fit=crop&crop=face',
    bio: 'Graduate student at Northwestern. Looking for someone to share a beautiful apartment with. Love hiking and reading.',
    isPetFriendly: false,
  },
];

const HomeScreen = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [profiles] = useState(sampleProfiles);
  
  // Animation values for swipe hints
  const leftOpacity = useRef(new Animated.Value(0.25)).current;
  const rightOpacity = useRef(new Animated.Value(0.25)).current;
  const pan = useRef(new Animated.ValueXY()).current;
  const scale = useRef(new Animated.Value(1)).current;

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (evt, gestureState) => {
        return Math.abs(gestureState.dx) > 20 || Math.abs(gestureState.dy) > 20;
      },
      onPanResponderGrant: () => {
        pan.setOffset({
          x: pan.x as unknown as number,
          y: pan.y as unknown as number,
        });
      },
      onPanResponderMove: (evt, gestureState) => {
        // Update pan animation
        pan.setValue({ x: gestureState.dx, y: gestureState.dy });
        
        // Calculate hint opacity based on drag distance
        const dragDistance = Math.abs(gestureState.dx);
        const maxDistance = screenWidth * 0.3;
        const opacity = Math.min(dragDistance / maxDistance, 1);
        
        if (gestureState.dx > 0) {
          // Dragging right - show connect hint
          rightOpacity.setValue(Math.max(0.25, opacity));
          leftOpacity.setValue(0.25);
        } else {
          // Dragging left - show skip hint
          leftOpacity.setValue(Math.max(0.25, opacity));
          rightOpacity.setValue(0.25);
        }
      },
      onPanResponderRelease: (evt, gestureState) => {
        const threshold = screenWidth * 0.25;
        
        if (Math.abs(gestureState.dx) > threshold) {
          // Card was swiped far enough
          const direction = gestureState.dx > 0 ? 'right' : 'left';
          handleSwipe(direction);
        } else {
          // Snap back to center
          Animated.spring(pan, {
            toValue: { x: 0, y: 0 },
            useNativeDriver: false,
          }).start();
        }
        
        // Reset hint opacities
        Animated.timing(leftOpacity, {
          toValue: 0.25,
          duration: 200,
          useNativeDriver: false,
        }).start();
        
        Animated.timing(rightOpacity, {
          toValue: 0.25,
          duration: 200,
          useNativeDriver: false,
        }).start();
        
        pan.flattenOffset();
      },
    })
  ).current;

  const handleSwipe = (direction: 'left' | 'right') => {
    const currentProfile = profiles[currentIndex];
    
    if (direction === 'right') {
      Alert.alert('Connected!', `You connected with ${currentProfile.name}`);
    } else {
      Alert.alert('Skipped', `You skipped ${currentProfile.name}`);
    }

    // Animate card out
    Animated.timing(pan, {
      toValue: { 
        x: direction === 'right' ? screenWidth : -screenWidth, 
        y: 0 
      },
      duration: 300,
      useNativeDriver: false,
    }).start(() => {
      // Move to next profile
      setCurrentIndex(prevIndex => {
        const nextIndex = prevIndex + 1;
        if (nextIndex >= profiles.length) {
          Alert.alert('No more profiles', 'You\'ve seen all available roommates!');
          return 0; // Reset to beginning for demo
        }
        return nextIndex;
      });
      
      // Reset animations
      pan.setValue({ x: 0, y: 0 });
      scale.setValue(1);
    });
  };

  const handleFilterPress = () => {
    Alert.alert('Filters', 'Filter functionality coming soon!');
  };

  const handleCardPress = () => {
    const currentProfile = profiles[currentIndex];
    Alert.alert(
      `${currentProfile.name}'s Profile`, 
      `Age: ${currentProfile.age}\nLocation: ${currentProfile.location}\nPrice: $${currentProfile.price}/month\n\n${currentProfile.bio}`
    );
  };

  const currentProfile = profiles[currentIndex];

  if (!currentProfile) {
    return null;
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="transparent" translucent />
      
      {/* Background Gradient */}
      <LinearGradient
        colors={colors.profileBackgroundGradient as [string, string]}
        style={StyleSheet.absoluteFillObject}
        start={{ x: 0, y: 0 }}
        end={{ x: 0, y: 1 }}
      />

      <SafeAreaView style={styles.safeArea}>
        {/* Header */}
        <HomeHeader onFilterPress={handleFilterPress} />

        {/* Card Stack Area */}
        <View style={styles.cardStackContainer}>
          {/* Swipeable Card */}
          <Animated.View
            style={[
              styles.cardWrapper,
              {
                transform: [
                  { translateX: pan.x },
                  { translateY: pan.y },
                  { scale: scale },
                ],
              },
            ]}
            {...panResponder.panHandlers}
          >
            <RoommateCard 
              profile={currentProfile}
              onCardPress={handleCardPress}
            />
          </Animated.View>

          {/* Background card (next profile) */}
          {profiles[currentIndex + 1] && (
            <View style={[styles.cardWrapper, styles.backgroundCard]}>
              <RoommateCard 
                profile={profiles[currentIndex + 1]}
                onCardPress={() => {}}
              />
            </View>
          )}

          {/* Swipe Hints */}
          <SwipeHints 
            leftOpacity={leftOpacity}
            rightOpacity={rightOpacity}
          />
        </View>
      </SafeAreaView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#E4F2FF', // Fallback color
  },
  safeArea: {
    flex: 1,
  },
  cardStackContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingBottom: 100, // Space for bottom navigation
  },
  cardWrapper: {
    position: 'absolute',
    width: screenWidth * 0.85,
    alignSelf: 'center',
  },
  backgroundCard: {
    transform: [{ scale: 0.95 }],
    opacity: 0.5,
    zIndex: 0,
  },
});

export default HomeScreen; 