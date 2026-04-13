import React, { useState } from 'react';
import { 
  View, 
  Text, 
  Image, 
  StyleSheet, 
  TouchableOpacity, 
  Dimensions 
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, fontSizes, shadows } from '../../styles/theme';

const { width: screenWidth } = Dimensions.get('window');

export interface RoommateProfile {
  id: string;
  name: string;
  age: number;
  location: string;
  price: number;
  bedrooms: string;
  gender: string;
  photo: string;
  bio?: string;
  isPetFriendly?: boolean;
}

interface RoommateCardProps {
  profile: RoommateProfile;
  onCardPress: () => void;
}

const RoommateCard = ({ profile, onCardPress }: RoommateCardProps) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <TouchableOpacity 
      style={styles.cardContainer}
      onPress={onCardPress}
      activeOpacity={0.95}
    >
      {/* Main card with image */}
      <View style={styles.card}>
        {/* Hero Image */}
        <Image 
          source={{ uri: profile.photo }} 
          style={styles.heroImage}
          resizeMode="cover"
        />
        
        {/* Price and bedroom chips overlay */}
        <View style={styles.chipsContainer}>
          <View style={styles.priceChip}>
            <Text style={styles.chipText}>${profile.price}/month</Text>
          </View>
          <View style={styles.bedroomChip}>
            <Text style={styles.chipText}>{profile.bedrooms}</Text>
          </View>
        </View>

        {/* Card footer */}
        <TouchableOpacity 
          style={styles.footer}
          onPress={toggleExpanded}
          activeOpacity={0.8}
        >
          <View style={styles.nameLocationContainer}>
            <Text style={styles.name}>
              {profile.name} {profile.age}
            </Text>
            <Text style={styles.location}>{profile.location}</Text>
          </View>
          
          <View style={styles.genderContainer}>
            <Ionicons 
              name={profile.gender === 'Male' ? 'male' : 'female'} 
              size={20} 
              color={colors.text.secondary} 
            />
            <Text style={styles.genderText}>{profile.gender}</Text>
            <Ionicons 
              name={isExpanded ? "chevron-up" : "chevron-down"} 
              size={20} 
              color={colors.text.secondary} 
              style={styles.expandIcon}
            />
          </View>

          {/* Expanded bio section */}
          {isExpanded && profile.bio && (
            <View style={styles.bioContainer}>
              <Text style={styles.bioText}>{profile.bio}</Text>
            </View>
          )}
        </TouchableOpacity>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  cardContainer: {
    width: screenWidth * 0.85,
    alignSelf: 'center',
  },
  card: {
    backgroundColor: colors.white,
    borderRadius: borderRadius.lg,
    ...shadows.medium,
    shadowOpacity: 0.08,
    overflow: 'hidden',
  },
  heroImage: {
    width: '100%',
    height: 400,
    backgroundColor: '#f0f0f0',
  },
  chipsContainer: {
    position: 'absolute',
    top: spacing.md,
    right: spacing.md,
    flexDirection: 'column',
    gap: spacing.sm,
  },
  priceChip: {
    backgroundColor: colors.primaryProfile,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: 20,
  },
  bedroomChip: {
    backgroundColor: colors.primaryProfile,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: 20,
  },
  chipText: {
    color: colors.white,
    fontSize: fontSizes.sm,
    fontWeight: '600',
  },
  footer: {
    backgroundColor: colors.white,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
  },
  nameLocationContainer: {
    marginBottom: spacing.xs,
  },
  name: {
    fontSize: fontSizes.xl,
    fontWeight: '600',
    color: colors.text.primary,
    marginBottom: spacing.xs / 2,
  },
  location: {
    fontSize: fontSizes.md,
    color: colors.text.secondary,
  },
  genderContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.sm,
  },
  genderText: {
    fontSize: fontSizes.sm,
    color: colors.text.secondary,
    marginLeft: spacing.xs,
  },
  expandIcon: {
    marginLeft: 'auto',
  },
  bioContainer: {
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  bioText: {
    fontSize: fontSizes.sm,
    color: colors.text.primary,
    lineHeight: fontSizes.sm * 1.4,
  },
});

export default RoommateCard; 