import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Image,
  TouchableOpacity,
  ScrollView,
  Dimensions,
  Modal,
  SafeAreaView,
} from 'react-native';
import { User } from '../models/User';
import { Ionicons } from '@expo/vector-icons';

interface DetailedProfileViewProps {
  user: User | null;
  visible: boolean;
  onClose: () => void;
  onLike?: (user: User) => void;
  onPass?: (user: User) => void;
}

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

const DetailedProfileView: React.FC<DetailedProfileViewProps> = ({
  user,
  visible,
  onClose,
  onLike,
  onPass,
}) => {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  if (!user) return null;

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

  const handleImageTap = () => {
    if (profileImages.length > 1) {
      const nextIndex = (currentImageIndex + 1) % profileImages.length;
      setCurrentImageIndex(nextIndex);
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

  const getCleanlinessLabel = (level: number) => {
    const labels = ['', 'Very Messy', 'Somewhat Messy', 'Average', 'Pretty Clean', 'Very Clean'];
    return labels[level] || 'Average';
  };

  const getNoiseLabel = (level: number) => {
    const labels = ['', 'Very Quiet', 'Quiet', 'Moderate', 'Lively', 'Very Lively'];
    return labels[level] || 'Moderate';
  };

  const getSocialLabel = (level: number) => {
    const labels = ['', 'Very Private', 'Private', 'Balanced', 'Social', 'Very Social'];
    return labels[level] || 'Balanced';
  };

  const formatRentBudget = (budget?: number) => {
    if (!budget) return '';
    return `$${budget}/month`;
  };

  return (
    <Modal visible={visible} animationType="slide" presentationStyle="pageSheet">
      <SafeAreaView style={styles.container}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity 
            style={styles.backButton} 
            onPress={onClose}
            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
          >
            <Ionicons name="chevron-back" size={28} color="#3498db" />
            <Text style={styles.backText}>Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Profile Details</Text>
          <View style={styles.headerSpacer} />
        </View>

        <ScrollView style={styles.scrollContainer} showsVerticalScrollIndicator={false}>
          {/* Profile Image */}
          <View style={styles.imageContainer}>
            <TouchableOpacity 
              style={styles.imageTouch}
              onPress={handleImageTap}
              activeOpacity={0.9}
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

          {/* Profile Info */}
          <View style={styles.profileInfo}>
            <Text style={styles.name}>{user.name}, {user.age}</Text>
            <Text style={styles.location}>
              <Ionicons name="location" size={16} color="#666" /> Chicago
            </Text>
            <Text style={styles.bio}>
              {user.descriptions && user.descriptions.length > 0 
                ? user.descriptions.join(' ') 
                : 'Jazz musician, vinyl collector, culinary student.'}
            </Text>
          </View>

          {/* Habits Section */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Habits</Text>
            <Text style={styles.sectionContent}>
              {user.habitsSummary || 'Late night practice sessions, cooking experiments, record collecting.'}
            </Text>
          </View>

          {/* Looking For Section */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Looking For</Text>
            <Text style={styles.sectionContent}>
              {user.lookingForSummary || 'Music lover who enjoys good food and late nights.'}
            </Text>
          </View>

          {/* Preferences Section */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Lifestyle Preferences</Text>
            
            <View style={styles.preferenceRow}>
              <View style={styles.preferenceItem}>
                <Ionicons name="sparkles" size={20} color="#3498db" />
                <Text style={styles.preferenceLabel}>Cleanliness</Text>
                <Text style={styles.preferenceValue}>{getCleanlinessLabel(user.preferences.cleanliness)}</Text>
              </View>
            </View>

            <View style={styles.preferenceRow}>
              <View style={styles.preferenceItem}>
                <Ionicons name="volume-medium" size={20} color="#3498db" />
                <Text style={styles.preferenceLabel}>Noise Level</Text>
                <Text style={styles.preferenceValue}>{getNoiseLabel(user.preferences.noiseLevel)}</Text>
              </View>
            </View>

            <View style={styles.preferenceRow}>
              <View style={styles.preferenceItem}>
                <Ionicons name="people" size={20} color="#3498db" />
                <Text style={styles.preferenceLabel}>Social Level</Text>
                <Text style={styles.preferenceValue}>{getSocialLabel(user.preferences.socialLevel)}</Text>
              </View>
            </View>

            <View style={styles.preferenceRow}>
              <View style={styles.preferenceItem}>
                <Ionicons name="time" size={20} color="#3498db" />
                <Text style={styles.preferenceLabel}>Sleep Schedule</Text>
                <Text style={styles.preferenceValue}>{getSleepScheduleLabel(user.preferences.sleepSchedule)}</Text>
              </View>
            </View>

            {user.preferences.monthlyRentBudget && (
              <View style={styles.preferenceRow}>
                <View style={styles.preferenceItem}>
                  <Ionicons name="cash" size={20} color="#27ae60" />
                  <Text style={styles.preferenceLabel}>Budget</Text>
                  <Text style={[styles.preferenceValue, styles.budgetValue]}>
                    {formatRentBudget(user.preferences.monthlyRentBudget)}
                  </Text>
                </View>
              </View>
            )}
          </View>

          {/* Property Photos Section */}
          {user.images?.property && user.images.property.length > 0 && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>
                <Ionicons name="home" size={20} color="#3498db" /> Property Photos
              </Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.propertyPhotos}>
                {user.images.property.map((imageUrl, index) => (
                  <Image
                    key={index}
                    source={{ uri: imageUrl }}
                    style={styles.propertyImage}
                  />
                ))}
              </ScrollView>
            </View>
          )}

          {/* Bottom spacing for action buttons */}
          <View style={styles.bottomSpacing} />
        </ScrollView>

        {/* Action Buttons */}
        <View style={styles.actionButtons}>
          <TouchableOpacity 
            style={[styles.actionButton, styles.passButton]}
            onPress={() => {
              onPass?.(user);
              onClose();
            }}
          >
            <Ionicons name="close" size={30} color="#fff" />
            <Text style={styles.actionButtonText}>Not Interested</Text>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={[styles.actionButton, styles.likeButton]}
            onPress={() => {
              onLike?.(user);
              onClose();
            }}
          >
            <Ionicons name="heart" size={30} color="#fff" />
            <Text style={styles.actionButtonText}>Like</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  backText: {
    color: '#3498db',
    fontSize: 17,
    marginLeft: 5,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  headerSpacer: {
    width: 60,
  },
  scrollContainer: {
    flex: 1,
  },
  imageContainer: {
    height: screenHeight * 0.6,
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
    top: 20,
    right: 20,
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
  profileInfo: {
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  name: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  location: {
    fontSize: 16,
    color: '#666',
    marginBottom: 16,
    flexDirection: 'row',
    alignItems: 'center',
  },
  bio: {
    fontSize: 16,
    color: '#555',
    lineHeight: 24,
  },
  section: {
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 16,
    flexDirection: 'row',
    alignItems: 'center',
  },
  sectionContent: {
    fontSize: 16,
    color: '#555',
    lineHeight: 24,
  },
  preferenceRow: {
    marginBottom: 16,
  },
  preferenceItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  preferenceLabel: {
    fontSize: 16,
    color: '#333',
    marginLeft: 12,
    flex: 1,
  },
  preferenceValue: {
    fontSize: 16,
    color: '#666',
    fontWeight: '500',
  },
  budgetValue: {
    color: '#27ae60',
    fontWeight: '600',
  },
  propertyPhotos: {
    marginTop: 12,
  },
  propertyImage: {
    width: 120,
    height: 90,
    borderRadius: 8,
    marginRight: 12,
    resizeMode: 'cover',
  },
  bottomSpacing: {
    height: 100,
  },
  actionButtons: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    paddingVertical: 20,
    paddingBottom: 30,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
  },
  actionButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 16,
    borderRadius: 25,
    marginHorizontal: 8,
  },
  passButton: {
    backgroundColor: '#ef4444',
  },
  likeButton: {
    backgroundColor: '#4ade80',
  },
  actionButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
});

export default DetailedProfileView; 