import React, { useState } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  Modal, 
  TouchableOpacity, 
  ScrollView, 
  Image, 
  Dimensions,
  FlatList
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Listing } from '../models/Listing';

interface PropertyDetailModalProps {
  listing: Listing | null;
  visible: boolean;
  onClose: () => void;
  onContactOwner: (listing: Listing) => void;
  onCreateGroup: (listing: Listing) => void;
}

const { width: screenWidth } = Dimensions.get('window');

const PropertyDetailModal: React.FC<PropertyDetailModalProps> = ({ 
  listing, 
  visible, 
  onClose,
  onContactOwner,
  onCreateGroup
}) => {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  if (!listing) return null;

  const { 
    address, 
    area, 
    zipCode, 
    homeDetails, 
    propertyImageUrls, 
    description 
  } = listing;

  const { 
    rooms, 
    bathrooms, 
    rent, 
    moveInDate, 
    leaseLength, 
    furnished, 
    petsAllowed, 
    amenities 
  } = homeDetails;

  // Format rent
  const formattedRent = `$${rent.toLocaleString()}/month`;

  // Format move-in date
  const formatDate = (timestamp: any) => {
    if (!timestamp) return 'Flexible';
    const date = timestamp.toDate();
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  // Next image
  const handleNextImage = () => {
    if (propertyImageUrls.length === 0) return;
    setCurrentImageIndex((currentImageIndex + 1) % propertyImageUrls.length);
  };

  return (
    <Modal
      visible={visible}
      animationType="slide"
      transparent={false}
      onRequestClose={onClose}
    >
      <View style={styles.container}>
        <TouchableOpacity style={styles.closeButton} onPress={onClose}>
          <Ionicons name="close" size={28} color="#333" />
        </TouchableOpacity>

        <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
          {/* Property Images */}
          <TouchableOpacity activeOpacity={0.9} onPress={handleNextImage}>
            <View style={styles.imageContainer}>
              {propertyImageUrls.length > 0 ? (
                <Image
                  source={{ uri: propertyImageUrls[currentImageIndex] }}
                  style={styles.propertyImage}
                  resizeMode="cover"
                />
              ) : (
                <View style={styles.noImageContainer}>
                  <Ionicons name="image-outline" size={50} color="#ccc" />
                  <Text style={styles.noImageText}>No photos available</Text>
                </View>
              )}
              
              {/* Image indicators */}
              {propertyImageUrls.length > 1 && (
                <View style={styles.imageIndicators}>
                  {propertyImageUrls.map((_, index) => (
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
          </TouchableOpacity>

          {/* Main Property Info */}
          <View style={styles.propertyInfo}>
            <Text style={styles.price}>{formattedRent}</Text>
            <Text style={styles.address}>{address}</Text>
            <Text style={styles.neighborhood}>{area}, Chicago {zipCode}</Text>

            {/* Quick Details */}
            <View style={styles.quickDetails}>
              <View style={styles.detailItem}>
                <Ionicons name="bed-outline" size={22} color="#666" />
                <Text style={styles.detailText}>{rooms} {rooms === 1 ? 'Room' : 'Rooms'}</Text>
              </View>
              <View style={styles.detailItem}>
                <Ionicons name="water-outline" size={22} color="#666" />
                <Text style={styles.detailText}>{bathrooms} {bathrooms === 1 ? 'Bath' : 'Baths'}</Text>
              </View>
              <View style={styles.detailItem}>
                <Ionicons name="calendar-outline" size={22} color="#666" />
                <Text style={styles.detailText}>{leaseLength} Mo Lease</Text>
              </View>
            </View>

            {/* Property Description */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Description</Text>
              <Text style={styles.description}>{description}</Text>
            </View>

            {/* Property Details */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Details</Text>
              
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Move-in Date:</Text>
                <Text style={styles.detailValue}>{formatDate(moveInDate)}</Text>
              </View>
              
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Furnished:</Text>
                <Text style={styles.detailValue}>{furnished ? 'Yes' : 'No'}</Text>
              </View>
              
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Pets Allowed:</Text>
                <Text style={styles.detailValue}>{petsAllowed ? 'Yes' : 'No'}</Text>
              </View>
            </View>

            {/* Amenities */}
            {amenities.length > 0 && (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Amenities</Text>
                <View style={styles.amenitiesContainer}>
                  {amenities.map((amenity, index) => (
                    <View key={index} style={styles.amenityTag}>
                      <Ionicons 
                        name={
                          amenity.includes('Laundry') ? 'water-outline' :
                          amenity.includes('Dishwasher') ? 'restaurant-outline' :
                          amenity.includes('AC') ? 'snow-outline' :
                          amenity.includes('Balcony') ? 'sunny-outline' :
                          amenity.includes('Gym') ? 'barbell-outline' :
                          amenity.includes('Pool') ? 'water-outline' :
                          amenity.includes('Parking') ? 'car-outline' :
                          'checkmark-circle-outline'
                        } 
                        size={16} 
                        color="#3498db" 
                      />
                      <Text style={styles.amenityText}>{amenity}</Text>
                    </View>
                  ))}
                </View>
              </View>
            )}

            {/* Action Buttons */}
            <View style={styles.actionButtons}>
              <TouchableOpacity 
                style={styles.actionButton} 
                onPress={() => onContactOwner(listing)}
              >
                <Ionicons name="mail-outline" size={20} color="#fff" />
                <Text style={styles.actionButtonText}>Contact Owner</Text>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={[styles.actionButton, styles.groupButton]} 
                onPress={() => onCreateGroup(listing)}
              >
                <Ionicons name="people-outline" size={20} color="#fff" />
                <Text style={styles.actionButtonText}>Create Group</Text>
              </TouchableOpacity>
            </View>
          </View>
        </ScrollView>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  closeButton: {
    position: 'absolute',
    top: 50,
    right: 20,
    zIndex: 10,
    backgroundColor: '#fff',
    borderRadius: 20,
    padding: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
  },
  scrollView: {
    flex: 1,
  },
  imageContainer: {
    width: '100%',
    height: 300,
    position: 'relative',
  },
  propertyImage: {
    width: '100%',
    height: '100%',
  },
  noImageContainer: {
    width: '100%',
    height: '100%',
    backgroundColor: '#f0f0f0',
    justifyContent: 'center',
    alignItems: 'center',
  },
  noImageText: {
    marginTop: 10,
    color: '#999',
    fontSize: 16,
  },
  imageIndicators: {
    position: 'absolute',
    bottom: 16,
    alignSelf: 'center',
    flexDirection: 'row',
    alignItems: 'center',
  },
  indicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginHorizontal: 4,
  },
  activeIndicator: {
    backgroundColor: '#3498db',
  },
  inactiveIndicator: {
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
  },
  propertyInfo: {
    padding: 20,
    paddingTop: 16,
  },
  price: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#3498db',
    marginBottom: 8,
  },
  address: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  neighborhood: {
    fontSize: 16,
    color: '#666',
    marginBottom: 16,
  },
  quickDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: '#eee',
    paddingVertical: 16,
    marginBottom: 20,
  },
  detailItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  detailText: {
    marginLeft: 8,
    fontSize: 16,
    color: '#666',
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  description: {
    fontSize: 16,
    lineHeight: 24,
    color: '#555',
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  detailLabel: {
    fontSize: 16,
    color: '#666',
  },
  detailValue: {
    fontSize: 16,
    color: '#333',
    fontWeight: '500',
  },
  amenitiesContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  amenityTag: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f0f8ff',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    marginRight: 8,
    marginBottom: 8,
  },
  amenityText: {
    fontSize: 14,
    color: '#3498db',
    marginLeft: 6,
  },
  actionButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 16,
    marginBottom: 30,
  },
  actionButton: {
    flex: 1,
    backgroundColor: '#3498db',
    borderRadius: 10,
    padding: 16,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 8,
  },
  groupButton: {
    backgroundColor: '#27ae60',
    marginRight: 0,
    marginLeft: 8,
  },
  actionButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
    marginLeft: 8,
  },
});

export default PropertyDetailModal; 