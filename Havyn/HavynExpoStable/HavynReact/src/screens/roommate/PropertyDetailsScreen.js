import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  TouchableOpacity,
  Linking,
  Platform,
  Dimensions,
  SafeAreaView,
  Alert
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../../utils/theme';
import ImagesCarousel from '../../components/ImagesCarousel';

const { width } = Dimensions.get('window');

const Amenity = ({ name, icon }) => (
  <View style={styles.amenityItem}>
    <Ionicons name={icon} size={20} color={COLORS.primary} />
    <Text style={styles.amenityText}>{name}</Text>
  </View>
);

const getAmenityIcon = (amenity) => {
  const amenityIcons = {
    'Pool': 'water-outline',
    'Gym': 'fitness-outline',
    'Doorman': 'person-outline',
    'Elevator': 'git-merge-outline',
    'Laundry': 'shirt-outline',
    'Dishwasher': 'water-outline',
    'In-unit Laundry': 'shirt-outline',
    'Parking': 'car-outline',
    'Fitness Center': 'fitness-outline',
    'Rooftop Deck': 'sunny-outline',
    'Hardwood Floors': 'grid-outline',
    'Pets Allowed': 'paw-outline',
    'Air Conditioning': 'snow-outline',
    'Parking Available': 'car-outline',
    'Balcony': 'home-outline'
  };
  
  return amenityIcons[amenity] || 'checkmark-circle-outline';
};

const PropertyDetailsScreen = ({ route, navigation }) => {
  const { property } = route.params;
  const [liked, setLiked] = useState(false);
  
  // Default coordinates for map preview
  const mapPreviewUrl = Platform.select({
    ios: `maps:0,0?q=${property.coordinate.latitude},${property.coordinate.longitude}`,
    android: `geo:0,0?q=${property.coordinate.latitude},${property.coordinate.longitude}`
  });
  
  const handleLike = () => {
    setLiked(!liked);
    if (!liked) {
      Alert.alert("Property Saved", "Property has been added to your favorites.");
    }
  };
  
  const handleContactOwner = () => {
    Alert.alert(
      "Contact Owner",
      `Would you like to reach out to ${property.profileName}?`,
      [
        { text: "Cancel", style: "cancel" },
        { text: "Message", onPress: () => Alert.alert("Message Sent", "Your interest has been sent to the owner. They will contact you soon!") },
        { text: "Call", onPress: () => Alert.alert("Call Feature", "In a real app, this would initiate a call to the owner.") }
      ]
    );
  };
  
  const openMap = () => {
    Linking.openURL(mapPreviewUrl).catch(() => {
      Alert.alert("Cannot Open Map", "Please make sure you have a map app installed.");
    });
  };
  
  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color={COLORS.text} />
        </TouchableOpacity>
        <View style={styles.headerRight}>
          <TouchableOpacity style={styles.actionButton} onPress={handleLike}>
            <Ionicons 
              name={liked ? "heart" : "heart-outline"} 
              size={24} 
              color={liked ? COLORS.primary : COLORS.text} 
            />
          </TouchableOpacity>
          <TouchableOpacity style={styles.actionButton}>
            <Ionicons name="share-social-outline" size={24} color={COLORS.text} />
          </TouchableOpacity>
        </View>
      </View>
      
      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Property Images */}
        <View style={styles.imagesContainer}>
          <ImagesCarousel images={property.images} />
          {property.verified && (
            <View style={styles.verifiedBadge}>
              <Ionicons name="checkmark-circle" size={16} color="#FFFFFF" />
              <Text style={styles.verifiedText}>Verified</Text>
            </View>
          )}
        </View>
        
        {/* Property Info */}
        <View style={styles.infoContainer}>
          <Text style={styles.propertyTitle}>{property.title}</Text>
          <Text style={styles.rentAmount}>{property.rent}</Text>
          
          <View style={styles.detailsRow}>
            <View style={styles.detailItem}>
              <Ionicons name="bed-outline" size={20} color={COLORS.primary} />
              <Text style={styles.detailText}>{property.bedrooms} {property.bedrooms === 1 ? 'Bedroom' : 'Bedrooms'}</Text>
            </View>
            <View style={styles.detailItem}>
              <Ionicons name="water-outline" size={20} color={COLORS.primary} />
              <Text style={styles.detailText}>{property.bathrooms} {property.bathrooms === 1 ? 'Bathroom' : 'Bathrooms'}</Text>
            </View>
            <View style={styles.detailItem}>
              <Ionicons name="location-outline" size={20} color={COLORS.primary} />
              <Text style={styles.detailText}>{property.neighborhood}</Text>
            </View>
          </View>
          
          {/* Owner Info */}
          <View style={styles.ownerContainer}>
            <View style={styles.ownerHeader}>
              <Text style={styles.sectionTitle}>Listed by</Text>
            </View>
            <View style={styles.ownerInfo}>
              <Image 
                source={property.images[0]} 
                style={styles.ownerImage} 
              />
              <View style={styles.ownerDetails}>
                <Text style={styles.ownerName}>{property.profileName}, {property.profileAge}</Text>
                <View style={styles.ownerVerification}>
                  <Ionicons name="shield-checkmark" size={16} color={COLORS.primary} />
                  <Text style={styles.verificationText}>Verified Owner</Text>
                </View>
              </View>
            </View>
          </View>
          
          {/* Description */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Description</Text>
            <Text style={styles.descriptionText}>{property.description}</Text>
          </View>
          
          {/* Property Features and Amenities */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Features & Amenities</Text>
            <View style={styles.amenitiesContainer}>
              {property.amenities.map((amenity, index) => (
                <Amenity key={index} name={amenity} icon={getAmenityIcon(amenity)} />
              ))}
            </View>
          </View>
          
          {/* Location */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Location</Text>
            <TouchableOpacity style={styles.mapPreview} onPress={openMap}>
              <View style={styles.mapImagePlaceholder}>
                <Ionicons name="map" size={60} color={COLORS.primary} style={styles.mapIcon} />
                <Text style={styles.mapText}>View in Maps</Text>
              </View>
              <Text style={styles.addressText}>{property.neighborhood}, Chicago, IL</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
      
      {/* Contact Owner Button */}
      <View style={styles.bottomBar}>
        <TouchableOpacity style={styles.contactButton} onPress={handleContactOwner}>
          <Text style={styles.contactButtonText}>Contact Owner</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 15,
    paddingVertical: 10,
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    zIndex: 10,
  },
  backButton: {
    padding: 8,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
  },
  headerRight: {
    flexDirection: 'row',
  },
  actionButton: {
    padding: 8,
    marginLeft: 10,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
  },
  imagesContainer: {
    width: '100%',
    height: 250,
    position: 'relative',
  },
  verifiedBadge: {
    position: 'absolute',
    top: 15,
    right: 15,
    backgroundColor: COLORS.primary,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 15,
  },
  verifiedText: {
    color: '#FFFFFF',
    marginLeft: 5,
    fontWeight: '600',
    fontSize: 12,
  },
  infoContainer: {
    padding: 20,
  },
  propertyTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 5,
  },
  rentAmount: {
    fontSize: 20,
    fontWeight: '700',
    color: COLORS.primary,
    marginBottom: 15,
  },
  detailsRow: {
    flexDirection: 'row',
    justifyContent: 'flex-start',
    marginBottom: 20,
    flexWrap: 'wrap',
  },
  detailItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 20,
    marginBottom: 10,
  },
  detailText: {
    fontSize: 14,
    marginLeft: 5,
    color: COLORS.textSecondary,
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: COLORS.text,
    marginBottom: 10,
  },
  descriptionText: {
    fontSize: 15,
    lineHeight: 22,
    color: COLORS.textSecondary,
  },
  amenitiesContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  amenityItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F0F6FF',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    marginRight: 10,
    marginBottom: 10,
  },
  amenityText: {
    fontSize: 14,
    marginLeft: 5,
    color: COLORS.text,
  },
  mapPreview: {
    width: '100%',
    borderRadius: 10,
    overflow: 'hidden',
    backgroundColor: '#f0f0f0',
  },
  mapImagePlaceholder: {
    width: '100%',
    height: 150,
    backgroundColor: '#e0e0e0',
    alignItems: 'center',
    justifyContent: 'center',
  },
  mapIcon: {
    opacity: 0.7,
  },
  mapText: {
    marginTop: 10,
    fontSize: 14,
    color: COLORS.primary,
    fontWeight: '600',
  },
  addressText: {
    fontSize: 14,
    padding: 10,
    color: COLORS.textSecondary,
  },
  ownerContainer: {
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#f0f0f0',
    borderRadius: 10,
    padding: 15,
  },
  ownerHeader: {
    marginBottom: 10,
  },
  ownerInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  ownerImage: {
    width: 50,
    height: 50,
    borderRadius: 25,
  },
  ownerDetails: {
    marginLeft: 15,
  },
  ownerName: {
    fontSize: 16,
    fontWeight: '600',
    color: COLORS.text,
  },
  ownerVerification: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 5,
  },
  verificationText: {
    fontSize: 13,
    color: COLORS.primary,
    marginLeft: 5,
  },
  bottomBar: {
    padding: 15,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
    backgroundColor: '#fff',
  },
  contactButton: {
    backgroundColor: COLORS.primary,
    paddingVertical: 15,
    borderRadius: 10,
    alignItems: 'center',
  },
  contactButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default PropertyDetailsScreen; 