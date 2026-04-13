import React, { useState } from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity, Dimensions, Animated } from 'react-native';
import { PanGestureHandler, State } from 'react-native-gesture-handler';
import { User } from '../models/User';
import { Ionicons } from '@expo/vector-icons';

interface ProfileCardProps {
  user: User;
  onPress?: () => void;
  onSwipeLeft?: (user: User) => void;
  onSwipeRight?: (user: User) => void;
  onMoreInfo?: (user: User) => void;
}

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');
const cardWidth = screenWidth - 24; // Bigger cards with less margin
const cardHeight = screenHeight * 0.60; // Adjusted for better visibility
const imageHeight = cardHeight * 0.60; // Reduced to 60% to ensure info section is fully visible
const SWIPE_THRESHOLD = 120;
const ROTATION_MULTIPLIER = 0.1;

const ProfileCard: React.FC<ProfileCardProps> = ({ 
  user, 
  onPress, 
  onSwipeLeft, 
  onSwipeRight, 
  onMoreInfo 
}) => {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  
  // Animation values for swipe gestures
  const translateX = new Animated.Value(0);
  const rotateZ = new Animated.Value(0);
  const scale = new Animated.Value(1);

  const getProfileTypeLabel = (type: string) => {
    switch (type) {
      case 'looking_for_room':
        return 'Looking for Room';
      case 'have_room':
        return 'Has Room Available';
      case 'apartment_listing':
        return 'Apartment Listing';
      default:
        return type;
    }
  };

  const getSleepScheduleLabel = (schedule: string) => {
    switch (schedule) {
      case 'early_bird':
        return 'Early Bird';
      case 'night_owl':
        return 'Night Owl';
      case 'regular':
        return 'Regular Schedule';
      default:
        return schedule;
    }
  };

  const getRatingStars = (rating: number) => {
    return '★'.repeat(rating) + '☆'.repeat(5 - rating);
  };

  const formatRentBudget = (budget?: number) => {
    if (!budget) return '';
    return `$${budget}/mo`;
  };

  // Get profile images (use new structure if available, fallback to legacy)
  const getProfileImages = (): string[] => {
    if (user.images?.profile && user.images.profile.length > 0) {
      return user.images.profile;
    }
    if (user.profileImageUrl) {
      return [user.profileImageUrl];
    }
    return ['https://via.placeholder.com/400x600/4a90e2/ffffff?text=No+Photo'];
  };

  const profileImages = getProfileImages();
  const budgetText = formatRentBudget(user.preferences.monthlyRentBudget);

  // Handle tap on image to cycle through photos
  const handleImageTap = () => {
    console.log('Image tapped! Current index:', currentImageIndex, 'Total images:', profileImages.length);
    if (profileImages.length > 1) {
      const nextIndex = (currentImageIndex + 1) % profileImages.length;
      console.log('Moving to index:', nextIndex, 'Image URL:', profileImages[nextIndex]);
      setCurrentImageIndex(nextIndex);
    } else {
      console.log('Only one image available, cannot cycle');
    }
  };

  const onGestureEvent = Animated.event(
    [{ nativeEvent: { translationX: translateX } }],
    { 
      useNativeDriver: true,
      listener: (event: any) => {
        const { translationX } = event.nativeEvent;
        
        // Update rotation based on translation
        rotateZ.setValue(translationX * ROTATION_MULTIPLIER);
        
        // Update scale based on translation (slight shrink when dragging)
        const scaleValue = 1 - Math.abs(translationX) / (screenWidth * 2);
        scale.setValue(Math.max(0.95, scaleValue));
      }
    }
  );

  const onHandlerStateChange = (event: any) => {
    const { state, translationX, velocityX } = event.nativeEvent;
    
    if (state === State.END) {
      const shouldSwipeRight = translationX > SWIPE_THRESHOLD || velocityX > 1000;
      const shouldSwipeLeft = translationX < -SWIPE_THRESHOLD || velocityX < -1000;
      
      if (shouldSwipeRight) {
        // Call callback immediately for instant transition
        onSwipeRight?.(user);
        
        // Run animation for visual feedback only
        Animated.parallel([
          Animated.timing(translateX, {
            toValue: screenWidth,
            duration: 100, // Ultra-fast animation
            useNativeDriver: true,
          }),
          Animated.timing(rotateZ, {
            toValue: screenWidth * ROTATION_MULTIPLIER,
            duration: 100,
            useNativeDriver: true,
          }),
          Animated.timing(scale, {
            toValue: 0.8,
            duration: 100,
            useNativeDriver: true,
          }),
        ]).start();
      } else if (shouldSwipeLeft) {
        // Call callback immediately for instant transition
        onSwipeLeft?.(user);
        
        // Run animation for visual feedback only
        Animated.parallel([
          Animated.timing(translateX, {
            toValue: -screenWidth,
            duration: 100, // Ultra-fast animation
            useNativeDriver: true,
          }),
          Animated.timing(rotateZ, {
            toValue: -screenWidth * ROTATION_MULTIPLIER,
            duration: 100,
            useNativeDriver: true,
          }),
          Animated.timing(scale, {
            toValue: 0.8,
            duration: 100,
            useNativeDriver: true,
          }),
        ]).start();
      } else {
        // Snap back to center
        Animated.parallel([
          Animated.spring(translateX, {
            toValue: 0,
            useNativeDriver: true,
          }),
          Animated.spring(rotateZ, {
            toValue: 0,
            useNativeDriver: true,
          }),
          Animated.spring(scale, {
            toValue: 1,
            useNativeDriver: true,
          }),
        ]).start();
      }
    }
  };

  const animatedStyle = {
    transform: [
      { translateX },
      { 
        rotateZ: rotateZ.interpolate({
          inputRange: [-screenWidth, screenWidth],
          outputRange: ['-20deg', '20deg'],
          extrapolate: 'clamp',
        })
      },
      { scale }
    ],
  };

  // Show swipe indicators based on translation
  const leftOpacity = translateX.interpolate({
    inputRange: [-screenWidth, -SWIPE_THRESHOLD, 0],
    outputRange: [1, 0.8, 0],
    extrapolate: 'clamp',
  });

  const rightOpacity = translateX.interpolate({
    inputRange: [0, SWIPE_THRESHOLD, screenWidth],
    outputRange: [0, 0.8, 1],
    extrapolate: 'clamp',
  });

  // Debug: Log profile images
  console.log('ProfileCard rendered for:', user.name);
  console.log('Available images:', profileImages);
  console.log('Current image index:', currentImageIndex);

  return (
    <PanGestureHandler
      onGestureEvent={onGestureEvent}
      onHandlerStateChange={onHandlerStateChange}
    >
      <Animated.View style={[styles.cardContainer, animatedStyle]}>
        {/* Swipe indicators */}
        <Animated.View style={[styles.swipeIndicator, styles.likeIndicator, { opacity: rightOpacity }]}>
          <Ionicons name="heart" size={40} color="#4ade80" />
          <Text style={styles.likeText}>LIKE</Text>
        </Animated.View>
        <Animated.View style={[styles.swipeIndicator, styles.passIndicator, { opacity: leftOpacity }]}>
          <Ionicons name="close" size={40} color="#ef4444" />
          <Text style={styles.passText}>PASS</Text>
        </Animated.View>

        <View style={styles.card}>
          <View style={styles.imageContainer}>
            <TouchableOpacity 
              style={styles.imageTouch}
              onPress={handleImageTap}
              activeOpacity={0.8}
            >
              <Image
                source={{ uri: profileImages[currentImageIndex] }}
                style={styles.profileImage}
                defaultSource={{ uri: 'https://via.placeholder.com/400x600/4a90e2/ffffff?text=No+Photo' }}
              />
            </TouchableOpacity>
            
            {/* Image indicators */}
            {profileImages.length > 1 && (
              <View style={styles.imageIndicators}>
                {profileImages.map((_, index) => (
                  <View
                    key={index}
                    style={[
                      styles.indicator,
                      currentImageIndex === index ? styles.activeIndicator : styles.inactiveIndicator
                    ]}
                  />
                ))}
              </View>
            )}
          </View>
          
          <View style={styles.infoContainer}>
            <View style={styles.nameAgeLocation}>
              <Text style={styles.name}>{user.name}, {user.age}</Text>
              <Text style={styles.location}>
                <Ionicons name="location-outline" size={16} color="#666" />  Chicago
              </Text>
            </View>

            <View style={styles.detailsContainer}>
              <View style={styles.tagsContainer}>
                {budgetText ? (
                  <View style={[styles.tag, styles.budgetTag]}>
                    <Ionicons name="cash-outline" size={14} color="#fff" />
                    <Text style={[styles.tagText, styles.budgetTagText]}>{budgetText}</Text>
                  </View>
                ) : null}

                <View style={styles.tag}>
                  <Ionicons name="time-outline" size={14} color="#333" />
                  <Text style={styles.tagText}>{getSleepScheduleLabel(user.preferences.sleepSchedule)}</Text>
                </View>

                {user.profileType === 'have_room' && (
                  <View style={[styles.tag, styles.primaryTag]}>
                    <Ionicons name="home-outline" size={14} color="white" />
                    <Text style={[styles.tagText, styles.primaryTagText]}>
                      {user.images?.property ? 'Room + Photos' : 'Has Room'}
                    </Text>
                  </View>
                )}
              </View>

              {/* Preview of description */}
              {user.descriptions && user.descriptions.length > 0 && (
                <View style={styles.descriptionPreview}>
                  <Text style={styles.descriptionText} numberOfLines={1}>
                    {user.descriptions[0]}
                  </Text>
                </View>
              )}

              <TouchableOpacity 
                style={styles.moreInfoButton}
                onPress={() => onMoreInfo?.(user)}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <Text style={styles.moreInfoText}>More Info</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Animated.View>
    </PanGestureHandler>
  );
};

const styles = StyleSheet.create({
  cardContainer: {
    position: 'relative',
    alignItems: 'center',
    marginBottom: 20, // Add bottom margin to ensure card is not cut off
  },
  card: {
    backgroundColor: 'white',
    borderRadius: 16,
    overflow: 'hidden',
    width: cardWidth,
    height: cardHeight,
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
  },
  swipeIndicator: {
    position: 'absolute',
    top: '25%',
    zIndex: 10,
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderRadius: 12,
    borderWidth: 3,
    alignItems: 'center',
    justifyContent: 'center',
  },
  likeIndicator: {
    right: 40,
    borderColor: '#4ade80',
    backgroundColor: 'rgba(74, 222, 128, 0.15)',
    transform: [{ rotate: '-15deg' }],
  },
  passIndicator: {
    left: 40,
    borderColor: '#ef4444',
    backgroundColor: 'rgba(239, 68, 68, 0.15)',
    transform: [{ rotate: '15deg' }],
  },
  likeText: {
    color: '#4ade80',
    fontSize: 18,
    fontWeight: 'bold',
    letterSpacing: 1,
    marginTop: 5,
  },
  passText: {
    color: '#ef4444',
    fontSize: 18,
    fontWeight: 'bold',
    letterSpacing: 1,
    marginTop: 5,
  },
  imageContainer: {
    height: imageHeight,
    width: '100%',
    backgroundColor: '#f0f0f0',
    position: 'relative',
  },
  imageTouch: {
    width: '100%',
    height: '100%',
  },
  profileImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  imageIndicators: {
    position: 'absolute',
    top: 16,
    right: 16,
    flexDirection: 'row',
    alignItems: 'center',
  },
  indicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginLeft: 4,
  },
  activeIndicator: {
    backgroundColor: 'white',
  },
  inactiveIndicator: {
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
  },
  infoContainer: {
    flex: 1,
    padding: 12,
    justifyContent: 'space-between',
    backgroundColor: 'white',
  },
  nameAgeLocation: {
    marginBottom: 4,
  },
  name: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 2,
  },
  location: {
    fontSize: 16,
    color: '#666',
    flexDirection: 'row',
    alignItems: 'center',
  },
  detailsContainer: {
    flex: 1,
    justifyContent: 'space-between',
  },
  moreInfoButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#3498db',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
    alignSelf: 'center',
    marginTop: 8,
    width: '60%',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  moreInfoText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '700',
    textAlign: 'center',
  },
  tagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginVertical: 4,
  },
  tag: {
    backgroundColor: '#f0f0f0',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 16,
    marginRight: 8,
    marginBottom: 4,
    flexDirection: 'row',
    alignItems: 'center',
  },
  budgetTag: {
    backgroundColor: '#27ae60',
  },
  primaryTag: {
    backgroundColor: '#3498db',
  },
  tagText: {
    fontSize: 12,
    color: '#333',
    fontWeight: '500',
    marginLeft: 4,
  },
  budgetTagText: {
    color: 'white',
  },
  primaryTagText: {
    color: 'white',
  },
  descriptionPreview: {
    marginVertical: 4,
  },
  descriptionText: {
    fontSize: 14,
    color: '#555',
    lineHeight: 18,
    fontStyle: 'italic',
  },
});

export default ProfileCard; 