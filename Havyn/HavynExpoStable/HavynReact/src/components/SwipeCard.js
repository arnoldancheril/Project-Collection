import React, { forwardRef, useImperativeHandle, useState, useEffect } from 'react';
import { View, StyleSheet, Text, Image, Dimensions, TouchableOpacity, Animated, Platform } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, SIZES, SHADOWS } from '../utils/theme';
import { LinearGradient } from 'expo-linear-gradient';
import { PanGestureHandler } from 'react-native-gesture-handler';
import Reanimated, { 
  useAnimatedGestureHandler, 
  useAnimatedStyle, 
  useSharedValue, 
  withSpring, 
  withTiming, 
  runOnJS,
  interpolate,
  Extrapolate
} from 'react-native-reanimated';

const { width, height } = Dimensions.get('window');
// Make card take up full screen with small padding
const CARD_WIDTH = width * 0.95;
const CARD_HEIGHT = height * 0.72; // Reduced height to avoid tab bar
const SWIPE_THRESHOLD = width / 3;

const AnimatedView = Reanimated.createAnimatedComponent(Animated.View);

// Helper function to format the price from different formats
const formatBudget = (profile) => {
  if (profile.budget) {
    return `$${profile.budget.min}-$${profile.budget.max}`;
  } else if (profile.property && profile.property.rent) {
    return profile.property.rent;
  }
  return 'Budget not specified';
};

// Helper function for neighborhood
const getNeighborhood = (profile) => {
  if (profile.location && profile.location.neighborhood) {
    return profile.location.neighborhood;
  } else if (profile.address && profile.address.neighborhood) {
    return profile.address.neighborhood;
  }
  return 'Location not specified';
};

// Helper function for amenities
const getAmenities = (profile) => {
  if (profile.property && profile.property.amenities) {
    return Array.isArray(profile.property.amenities) 
      ? profile.property.amenities.join(', ')
      : profile.property.amenities;
  }
  return '';
};

// Helper function to format move-in date
const formatDate = (date) => {
  if (!date) return "Flexible";
  
  const options = { month: 'numeric', day: 'numeric', year: 'numeric' };
  return new Date(date).toLocaleDateString(undefined, options);
};

const SwipeCard = forwardRef(({ profile, onSwipeLeft, onSwipeRight, onPress, isActive = true, style = {} }, ref) => {
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const scale = useSharedValue(1);
  const cardOpacity = useSharedValue(1);
  const likeOpacity = useSharedValue(0);
  const passOpacity = useSharedValue(0);
  const [isCardActive, setIsCardActive] = useState(isActive);

  // Reset animation values when new profile is loaded
  useEffect(() => {
    translateX.value = 0;
    translateY.value = 0;
    scale.value = 1;
    cardOpacity.value = 1;
    likeOpacity.value = 0;
    passOpacity.value = 0;
    setIsCardActive(isActive);
  }, [profile, isActive]);

  const gestureHandler = useAnimatedGestureHandler({
    onStart: (_, ctx) => {
      ctx.startX = translateX.value;
      ctx.startY = translateY.value;
    },
    onActive: (event, ctx) => {
      if (!isActive) return;
      
      translateX.value = ctx.startX + event.translationX;
      translateY.value = ctx.startY + event.translationY;
      
      // Show like/pass overlay based on swipe direction
      if (translateX.value > 20) {
        likeOpacity.value = interpolate(
          translateX.value,
          [0, SWIPE_THRESHOLD],
          [0, 1],
          Extrapolate.CLAMP
        );
        passOpacity.value = 0;
      } else if (translateX.value < -20) {
        passOpacity.value = interpolate(
          translateX.value,
          [-SWIPE_THRESHOLD, 0],
          [1, 0],
          Extrapolate.CLAMP
        );
        likeOpacity.value = 0;
      }
    },
    onEnd: (event) => {
      if (!isActive) return;
      
      if (translateX.value > SWIPE_THRESHOLD) {
        // Swiped right - LIKE
        translateX.value = withTiming(width * 1.5, { duration: 250 });
        translateY.value = withTiming(50, { duration: 250 });
        scale.value = withTiming(0.8, { duration: 250 });
        cardOpacity.value = withTiming(0, { duration: 200 }, () => {
          runOnJS(setIsCardActive)(false);
          runOnJS(onSwipeRight)();
        });
      } else if (translateX.value < -SWIPE_THRESHOLD) {
        // Swiped left - PASS
        translateX.value = withTiming(-width * 1.5, { duration: 250 });
        translateY.value = withTiming(50, { duration: 250 });
        scale.value = withTiming(0.8, { duration: 250 });
        cardOpacity.value = withTiming(0, { duration: 200 }, () => {
          runOnJS(setIsCardActive)(false);
          runOnJS(onSwipeLeft)();
        });
      } else {
        // Reset position if not swiped enough
        translateX.value = withSpring(0);
        translateY.value = withSpring(0);
        likeOpacity.value = withTiming(0);
        passOpacity.value = withTiming(0);
      }
    },
  });

  const cardStyle = useAnimatedStyle(() => {
    return {
      transform: [
        { translateX: translateX.value },
        { translateY: translateY.value },
        { scale: scale.value },
        { 
          rotate: `${interpolate(
            translateX.value, 
            [-SWIPE_THRESHOLD * 2, 0, SWIPE_THRESHOLD * 2], 
            [-30, 0, 30]
          )}deg` 
        }
      ],
      opacity: cardOpacity.value,
    };
  });

  const likeTextStyle = useAnimatedStyle(() => {
    return {
      opacity: likeOpacity.value,
      transform: [
        { scale: interpolate(likeOpacity.value, [0, 1], [0.8, 1.5]) }
      ]
    };
  });

  const passTextStyle = useAnimatedStyle(() => {
    return {
      opacity: passOpacity.value,
      transform: [
        { scale: interpolate(passOpacity.value, [0, 1], [0.8, 1.5]) }
      ]
    };
  });

  const swipeLeft = () => {
    if (!isCardActive) return;
    
    translateX.value = withTiming(-width * 1.5, { duration: 250 });
    translateY.value = withTiming(50, { duration: 250 });
    scale.value = withTiming(0.8, { duration: 250 });
    passOpacity.value = withTiming(1, { duration: 100 });
    cardOpacity.value = withTiming(0, { duration: 200 }, () => {
      runOnJS(setIsCardActive)(false);
      runOnJS(onSwipeLeft)();
    });
  };

  const swipeRight = () => {
    if (!isCardActive) return;
    
    translateX.value = withTiming(width * 1.5, { duration: 250 });
    translateY.value = withTiming(50, { duration: 250 });
    scale.value = withTiming(0.8, { duration: 250 });
    likeOpacity.value = withTiming(1, { duration: 100 });
    cardOpacity.value = withTiming(0, { duration: 200 }, () => {
      runOnJS(setIsCardActive)(false);
      runOnJS(onSwipeRight)();
    });
  };

  useImperativeHandle(ref, () => ({
    swipeLeft,
    swipeRight,
  }));

  // Check for age in profile and handle null/undefined
  const displayAge = profile.age ? profile.age.toString() : "??";

  if (!profile) {
    return (
      <View style={[styles.card, styles.emptyCard]}>
        <Text style={styles.emptyText}>No more profiles to show!</Text>
        <Text style={styles.emptySubText}>Check back later or adjust your preferences.</Text>
      </View>
    );
  }

  // Modern, simplified card view
  return (
    <PanGestureHandler onGestureEvent={gestureHandler} enabled={isCardActive && isActive}>
      <AnimatedView style={[styles.card, cardStyle, style]}>
        <TouchableOpacity 
          style={styles.cardContainer}
          activeOpacity={0.9}
          onPress={onPress}
        >
          <Image 
            source={profile.images && profile.images.length > 0 ? profile.images[0] : require('../../assets/person-placeholder.jpg')} 
            style={styles.image} 
            resizeMode="cover"
          />
          
          {/* Gradient overlay for better text readability */}
          <LinearGradient
            colors={['transparent', 'rgba(0,0,0,0.7)']}
            style={styles.gradient}
          />

          {/* Simplified profile info overlay */}
          <View style={styles.profileOverlay}>
            <View style={styles.nameRow}>
              <Text style={styles.nameText}>
                {profile.firstName || "Anonymous"}{", "}
                {displayAge}
              </Text>
              <View style={styles.locationBadge}>
                <Text style={styles.locationText}>
                  {getNeighborhood(profile)}
                </Text>
              </View>
            </View>
            
            <Text style={styles.bioText} numberOfLines={2}>
              {profile.bio || "No bio available"}
            </Text>
            
            <View style={styles.quickInfoContainer}>
              <View style={styles.quickInfoItem}>
                <Ionicons name="briefcase" size={18} color="#fff" />
                <Text style={styles.quickInfoText}>{profile.occupation || "Not specified"}</Text>
              </View>
              
              <View style={styles.quickInfoItem}>
                <Ionicons name="cash" size={18} color="#fff" />
                <Text style={styles.quickInfoText}>{formatBudget(profile)}</Text>
              </View>
            </View>
          </View>

          {/* "More Info" button positioned at bottom center, away from tab bar */}
          <TouchableOpacity 
            style={styles.moreInfoButton}
            onPress={onPress}
          >
            <Text style={styles.moreInfoText}>More Info</Text>
            <Ionicons name="chevron-forward" size={18} color="#fff" />
          </TouchableOpacity>

          {/* Like/Pass indicators - enhanced for better visibility */}
          <Reanimated.View style={[styles.indicator, styles.likeIndicator, likeTextStyle]}>
            <Ionicons name="heart" size={36} color={COLORS.primary} />
            <Text style={styles.indicatorText}>LIKE</Text>
          </Reanimated.View>
          
          <Reanimated.View style={[styles.indicator, styles.dislikeIndicator, passTextStyle]}>
            <Ionicons name="close" size={36} color={COLORS.secondary} />
            <Text style={styles.indicatorText}>PASS</Text>
          </Reanimated.View>
        </TouchableOpacity>
      </AnimatedView>
    </PanGestureHandler>
  );
});

const styles = StyleSheet.create({
  card: {
    position: 'absolute',
    width: CARD_WIDTH,
    height: CARD_HEIGHT,
    borderRadius: 20,
    overflow: 'hidden',
    backgroundColor: '#FFFFFF',
    ...SHADOWS.medium,
  },
  cardContainer: {
    flex: 1,
  },
  image: {
    width: '100%',
    height: '100%',
    position: 'absolute',
  },
  gradient: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    height: '60%',
    borderRadius: 20,
  },
  profileOverlay: {
    position: 'absolute',
    bottom: 80, // Provide space for the "More Info" button
    left: 0,
    right: 0,
    padding: 20,
  },
  nameRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  nameText: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textShadowColor: 'rgba(0, 0, 0, 0.3)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 3,
  },
  locationBadge: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 15,
  },
  locationText: {
    fontSize: 14,
    color: '#FFFFFF',
    fontWeight: '600',
  },
  bioText: {
    fontSize: 16,
    color: '#FFFFFF',
    textShadowColor: 'rgba(0, 0, 0, 0.3)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 3,
    marginBottom: 12,
  },
  quickInfoContainer: {
    marginTop: 8,
  },
  quickInfoItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
  },
  quickInfoText: {
    fontSize: 16,
    color: '#FFFFFF',
    marginLeft: 8,
    textShadowColor: 'rgba(0, 0, 0, 0.3)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 2,
  },
  moreInfoButton: {
    position: 'absolute',
    bottom: 25, // Positioned away from tab bar
    alignSelf: 'center',
    backgroundColor: COLORS.primary,
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 25,
    flexDirection: 'row',
    alignItems: 'center',
    ...SHADOWS.medium,
  },
  moreInfoText: {
    fontSize: 16,
    color: 'white',
    fontWeight: '600',
    marginRight: 5,
  },
  indicator: {
    position: 'absolute',
    paddingHorizontal: 25,
    paddingVertical: 15,
    borderRadius: 12,
    borderWidth: 4,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
    flexDirection: 'row',
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    top: '25%', // Positioned higher on the card
    ...SHADOWS.large,
  },
  likeIndicator: {
    left: '5%',  // Positioned on the left side
    borderColor: COLORS.primary,
  },
  dislikeIndicator: {
    right: '5%', // Positioned on the right side
    borderColor: COLORS.secondary,
  },
  indicatorText: {
    fontSize: 32,
    fontWeight: 'bold',
    marginLeft: 10,
  },
  emptyCard: {
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  emptyText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: COLORS.text,
    textAlign: 'center',
    marginBottom: 10,
  },
  emptySubText: {
    fontSize: 16,
    color: COLORS.textSecondary,
    textAlign: 'center',
  },
});

export default SwipeCard; 