import React, { useState } from 'react';
import {
  View,
  Text,
  Image,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
  FlatList,
  StatusBar
} from 'react-native';
import { Ionicons, FontAwesome5, MaterialCommunityIcons } from '@expo/vector-icons';
import { COLORS, SHADOWS } from '../../utils/theme';
import { useProfile } from '../../contexts/ProfileContext';

const { width, height } = Dimensions.get('window');

const DetailedProfileScreen = ({ route, navigation }) => {
  // Destructure the profile from params and ensure dates are serialized
  const { profile: originalProfile } = route.params;
  
  // Create a serializable copy of the profile with proper date handling
  const profile = React.useMemo(() => {
    const profileCopy = {...originalProfile};
    
    // Convert Date objects to strings for serializability
    if (profileCopy.moveInDate && profileCopy.moveInDate instanceof Date) {
      profileCopy.moveInDate = profileCopy.moveInDate.toISOString();
    }
    
    return profileCopy;
  }, [originalProfile]);
  
  const [activeTab, setActiveTab] = useState('about');
  const [activeImageIndex, setActiveImageIndex] = useState(0);
  const { likeProfile, dislikeProfile, likedProfiles } = useProfile();
  
  // Check if this profile is already liked
  const isLiked = likedProfiles.some(p => p.id === profile.id);

  const handleLikeProfile = () => {
    likeProfile(profile);
    navigation.goBack();
  };

  const handleDislikeProfile = () => {
    dislikeProfile(profile);
    navigation.goBack();
  };

  // Mock images in case the profile doesn't have multiple images
  const images = profile.images && profile.images.length > 0 
    ? profile.images 
    : [require('../../../assets/person-placeholder.jpg')];

  const renderImageIndicator = () => {
    return (
      <View style={styles.indicatorContainer}>
        {images.map((_, index) => (
          <View
            key={index}
            style={[
              styles.indicator,
              activeImageIndex === index && styles.activeIndicator,
            ]}
          />
        ))}
      </View>
    );
  };

  // Add safety for habits data
  const habits = profile.habits || 
    (profile.lifestylePreferences 
      ? Object.entries(profile.lifestylePreferences).map(([key, value]) => `${key}: ${value}`).join(', ')
      : 'Not specified');

  // Format amenities
  const amenities = profile.property && profile.property.amenities 
    ? (Array.isArray(profile.property.amenities) 
        ? profile.property.amenities 
        : profile.property.amenities.split(',').map(a => a.trim()))
    : [];

  // Get move-in date - handle different formats
  const moveInDate = profile.moveInDate 
    ? (typeof profile.moveInDate === 'string'
        ? new Date(profile.moveInDate).toLocaleDateString() 
        : profile.moveInDate.toLocaleDateString())
    : 'Flexible';

  // Tab content components
  const AboutTab = () => (
    <View style={styles.tabContent}>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Bio</Text>
        <Text style={styles.bioText}>{profile.bio || "No bio available"}</Text>
      </View>
      
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Occupation</Text>
        <View style={styles.infoRow}>
          <FontAwesome5 name="briefcase" size={20} color={COLORS.primary} />
          <Text style={styles.infoText}>{profile.occupation || "Not specified"}</Text>
        </View>
      </View>
      
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Location</Text>
        <View style={styles.infoRow}>
          <Ionicons name="location" size={20} color={COLORS.primary} />
          <Text style={styles.infoText}>
            {profile.location?.city || "Unknown"}, {profile.location?.state || ""}
            {profile.location?.neighborhood ? ` (${profile.location.neighborhood})` : ""}
          </Text>
        </View>
      </View>
    </View>
  );
  
  const PreferencesTab = () => (
    <View style={styles.tabContent}>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Lifestyle & Habits</Text>
        {profile.lifestylePreferences ? (
          Object.entries(profile.lifestylePreferences).map(([key, value], index) => (
            <View key={index} style={styles.preferenceItem}>
              <Text style={styles.preferenceLabel}>{key.charAt(0).toUpperCase() + key.slice(1)}</Text>
              <Text style={styles.preferenceValue}>{value}</Text>
            </View>
          ))
        ) : (
          <Text style={styles.noDataText}>No lifestyle preferences specified</Text>
        )}
      </View>
      
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Looking For</Text>
        <Text style={styles.infoText}>{profile.lookingFor || "Not specified"}</Text>
      </View>
      
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Budget Range</Text>
        <View style={styles.infoRow}>
          <Ionicons name="cash" size={20} color={COLORS.primary} />
          <Text style={styles.infoText}>
            {profile.budget 
              ? `$${profile.budget.min}-$${profile.budget.max}` 
              : (profile.property?.rent || "Not specified")}
          </Text>
        </View>
      </View>
      
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Move-in Date</Text>
        <View style={styles.infoRow}>
          <Ionicons name="calendar" size={20} color={COLORS.primary} />
          <Text style={styles.infoText}>{moveInDate}</Text>
        </View>
      </View>
    </View>
  );
  
  const SpaceTab = () => (
    <View style={styles.tabContent}>
      {profile.property ? (
        <>
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Property Details</Text>
            <View style={styles.infoRow}>
              <Ionicons name="home" size={20} color={COLORS.primary} />
              <Text style={styles.infoText}>{profile.property.type || "Not specified"}</Text>
            </View>
            
            <View style={styles.infoRow}>
              <Ionicons name="bed" size={20} color={COLORS.primary} />
              <Text style={styles.infoText}>
                {profile.property.rooms || "?"} Bedroom{profile.property.rooms !== 1 ? "s" : ""}, {" "}
                {profile.property.bathrooms || "?"} Bathroom{profile.property.bathrooms !== 1 ? "s" : ""}
              </Text>
            </View>
            
            <View style={styles.infoRow}>
              <Ionicons name="cash" size={20} color={COLORS.primary} />
              <Text style={styles.infoText}>{profile.property.rent || "Not specified"}</Text>
            </View>
            
            <View style={styles.infoRow}>
              <Ionicons name="location" size={20} color={COLORS.primary} />
              <Text style={styles.infoText}>{profile.property.address || "Address not provided"}</Text>
            </View>
          </View>
          
          {amenities.length > 0 && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Amenities</Text>
              <View style={styles.amenitiesContainer}>
                {amenities.map((amenity, index) => (
                  <View key={index} style={styles.amenityItem}>
                    <Ionicons name="checkmark-circle" size={16} color={COLORS.primary} />
                    <Text style={styles.amenityText}>{amenity}</Text>
                  </View>
                ))}
              </View>
            </View>
          )}
          
          {profile.property.coordinate && (
            <TouchableOpacity 
              style={styles.mapButton}
              onPress={() => navigation.navigate('Map', { initialLocation: profile.property.coordinate })}
            >
              <Ionicons name="map" size={18} color="#fff" />
              <Text style={styles.mapButtonText}>View on Map</Text>
            </TouchableOpacity>
          )}
        </>
      ) : (
        <Text style={styles.noDataText}>No property details available</Text>
      )}
    </View>
  );

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      {/* Image Gallery */}
      <View style={styles.imageContainer}>
        <FlatList
          data={images}
          horizontal
          pagingEnabled
          showsHorizontalScrollIndicator={false}
          keyExtractor={(_, index) => index.toString()}
          onMomentumScrollEnd={(event) => {
            const slideIndex = Math.floor(
              event.nativeEvent.contentOffset.x / width
            );
            setActiveImageIndex(slideIndex);
          }}
          renderItem={({ item }) => (
            <Image source={item} style={styles.profileImage} resizeMode="cover" />
          )}
        />
        {renderImageIndicator()}
        
        {/* Back button */}
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Ionicons name="chevron-back" size={28} color="#fff" />
        </TouchableOpacity>
      </View>

      {/* Profile Header */}
      <View style={styles.headerContainer}>
        <View style={styles.header}>
          <View>
            <Text style={styles.nameText}>
              {profile.firstName || profile.name || "Anonymous"}, {profile.age || "??"}
            </Text>
            <Text style={styles.locationText}>
              {profile.location?.city || "Unknown"}, {profile.location?.state || ""}
            </Text>
          </View>
          
          {/* Quick info pills */}
          <View style={styles.pillsContainer}>
            <View style={[styles.pill, { backgroundColor: COLORS.primary + '20' }]}>
              <Ionicons name="briefcase" size={14} color={COLORS.primary} />
              <Text style={styles.pillText}>{profile.occupation || "Unknown"}</Text>
            </View>
            
            {profile.budget && (
              <View style={[styles.pill, { backgroundColor: COLORS.secondary + '20' }]}>
                <Ionicons name="cash" size={14} color={COLORS.secondary} />
                <Text style={styles.pillText}>
                  ${profile.budget.min}-${profile.budget.max}
                </Text>
              </View>
            )}
          </View>
        </View>
        
        {/* Tab navigation */}
        <View style={styles.tabBar}>
          <TouchableOpacity 
            style={[styles.tab, activeTab === 'about' && styles.activeTab]}
            onPress={() => setActiveTab('about')}
          >
            <Text style={[styles.tabText, activeTab === 'about' && styles.activeTabText]}>About</Text>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={[styles.tab, activeTab === 'preferences' && styles.activeTab]}
            onPress={() => setActiveTab('preferences')}
          >
            <Text style={[styles.tabText, activeTab === 'preferences' && styles.activeTabText]}>Preferences</Text>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={[styles.tab, activeTab === 'space' && styles.activeTab]}
            onPress={() => setActiveTab('space')}
          >
            <Text style={[styles.tabText, activeTab === 'space' && styles.activeTabText]}>Space</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Tab content */}
      <ScrollView style={styles.contentContainer} showsVerticalScrollIndicator={false}>
        {activeTab === 'about' && <AboutTab />}
        {activeTab === 'preferences' && <PreferencesTab />}
        {activeTab === 'space' && <SpaceTab />}
        
        {/* Add bottom padding for scrolling and action buttons */}
        <View style={{ height: 100 }} />
      </ScrollView>

      {/* Action Buttons */}
      <View style={styles.actionButtonsContainer}>
        <TouchableOpacity
          style={[styles.actionButton, styles.dislikeButton]}
          onPress={handleDislikeProfile}
        >
          <Ionicons name="close" size={28} color="#fff" />
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[styles.actionButton, styles.likeButton]}
          onPress={handleLikeProfile}
        >
          <Ionicons name={isLiked ? "heart" : "heart-outline"} size={28} color="#FFFFFF" />
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  imageContainer: {
    height: height * 0.45,
    width: width,
    position: 'relative',
  },
  profileImage: {
    width: width,
    height: '100%',
  },
  indicatorContainer: {
    flexDirection: 'row',
    position: 'absolute',
    bottom: 20,
    alignSelf: 'center',
  },
  indicator: {
    height: 8,
    width: 8,
    borderRadius: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.6)',
    marginHorizontal: 4,
  },
  activeIndicator: {
    backgroundColor: COLORS.primary,
    width: 20,
  },
  backButton: {
    position: 'absolute',
    top: 50,
    left: 20,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 10,
  },
  headerContainer: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 30,
    borderTopRightRadius: 30,
    marginTop: -30,
    paddingTop: 20,
    paddingHorizontal: 20,
    ...SHADOWS.medium,
  },
  header: {
    marginBottom: 20,
  },
  nameText: {
    fontSize: 26,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 4,
  },
  locationText: {
    fontSize: 16,
    color: COLORS.textSecondary,
    marginBottom: 12,
  },
  pillsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 5,
  },
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    marginRight: 8,
    marginBottom: 8,
  },
  pillText: {
    fontSize: 12,
    fontWeight: '600',
    marginLeft: 4,
  },
  tabBar: {
    flexDirection: 'row',
    marginTop: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  tab: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
  },
  activeTab: {
    borderBottomWidth: 2,
    borderBottomColor: COLORS.primary,
  },
  tabText: {
    fontSize: 15,
    fontWeight: '500',
    color: COLORS.textSecondary,
  },
  activeTabText: {
    color: COLORS.primary,
    fontWeight: '600',
  },
  contentContainer: {
    flex: 1,
    backgroundColor: '#f8f9fa',
    paddingTop: 15,
  },
  tabContent: {
    paddingHorizontal: 20,
  },
  section: {
    backgroundColor: '#fff',
    borderRadius: 15,
    padding: 16,
    marginBottom: 16,
    ...SHADOWS.small,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 12,
  },
  bioText: {
    fontSize: 15,
    color: COLORS.textSecondary,
    lineHeight: 22,
  },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  infoText: {
    fontSize: 15,
    color: COLORS.textSecondary,
    marginLeft: 10,
    flex: 1,
  },
  preferenceItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  preferenceLabel: {
    fontSize: 15,
    color: COLORS.text,
  },
  preferenceValue: {
    fontSize: 15,
    color: COLORS.primary,
    fontWeight: '500',
  },
  amenitiesContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  amenityItem: {
    flexDirection: 'row',
    alignItems: 'center',
    width: '50%',
    marginBottom: 10,
  },
  amenityText: {
    fontSize: 14,
    color: COLORS.textSecondary,
    marginLeft: 6,
  },
  mapButton: {
    backgroundColor: COLORS.primary,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: 10,
    marginTop: 10,
  },
  mapButtonText: {
    color: '#fff',
    fontWeight: '600',
    marginLeft: 8,
  },
  noDataText: {
    fontSize: 15,
    color: COLORS.textSecondary,
    fontStyle: 'italic',
  },
  actionButtonsContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    paddingVertical: 15,
    position: 'absolute',
    bottom: 20,
    left: 0,
    right: 0,
  },
  actionButton: {
    width: 60,
    height: 60,
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
    marginHorizontal: 15,
    ...SHADOWS.medium,
  },
  dislikeButton: {
    backgroundColor: COLORS.secondary,
  },
  likeButton: {
    backgroundColor: COLORS.primary,
  },
});

export default DetailedProfileScreen; 