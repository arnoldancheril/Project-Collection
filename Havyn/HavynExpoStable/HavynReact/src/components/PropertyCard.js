import React from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity, Dimensions } from 'react-native';
import { Ionicons, MaterialIcons } from '@expo/vector-icons';
import { COLORS, SIZES, SHADOWS } from '../utils/theme';
import { formatPrice } from '../utils/helpers';

const { width } = Dimensions.get('window');
const CARD_WIDTH = width * 0.85;

const PropertyCard = ({ property, onPress, horizontal = false }) => {
  if (!property) return null;
  
  return (
    <TouchableOpacity
      style={[
        styles.container,
        horizontal ? styles.horizontalContainer : styles.verticalContainer,
        SHADOWS.medium
      ]}
      onPress={onPress}
      activeOpacity={0.8}
    >
      <Image
        source={{ uri: property.photos[0] }}
        style={horizontal ? styles.horizontalImage : styles.verticalImage}
        resizeMode="cover"
      />
      
      <View style={styles.contentContainer}>
        <View style={styles.headerRow}>
          <Text style={styles.priceText}>{formatPrice(property.price)}</Text>
          <View style={styles.statusContainer}>
            <View style={[styles.statusBadge, styles[`${property.status}Badge`]]}>
              <Text style={styles.statusText}>{property.status.toUpperCase()}</Text>
            </View>
          </View>
        </View>
        
        <Text style={styles.titleText} numberOfLines={1}>{property.title}</Text>
        
        <View style={styles.addressRow}>
          <Ionicons name="location-outline" size={14} color={COLORS.textSecondary} />
          <Text style={styles.addressText} numberOfLines={1}>
            {property.address.street}, {property.address.city}, {property.address.state}
          </Text>
        </View>
        
        <View style={styles.featuresRow}>
          <View style={styles.featureItem}>
            <MaterialIcons name="bed" size={16} color={COLORS.textSecondary} />
            <Text style={styles.featureText}>{property.bedrooms} bd</Text>
          </View>
          
          <View style={styles.featureItem}>
            <MaterialIcons name="bathtub" size={16} color={COLORS.textSecondary} />
            <Text style={styles.featureText}>{property.bathrooms} ba</Text>
          </View>
          
          <View style={styles.featureItem}>
            <MaterialIcons name="home" size={16} color={COLORS.textSecondary} />
            <Text style={styles.featureText}>{property.squareFeet} sqft</Text>
          </View>
          
          {property.availableRooms > 0 && (
            <View style={styles.featureItem}>
              <MaterialIcons name="meeting-room" size={16} color={COLORS.primary} />
              <Text style={[styles.featureText, styles.highlightText]}>
                {property.availableRooms} room{property.availableRooms > 1 ? 's' : ''} available
              </Text>
            </View>
          )}
        </View>
        
        <View style={styles.amenitiesContainer}>
          {property.amenities.slice(0, 3).map((amenity, index) => (
            <View key={index} style={styles.amenityTag}>
              <Text style={styles.amenityText}>{amenity}</Text>
            </View>
          ))}
          {property.amenities.length > 3 && (
            <View style={styles.amenityTag}>
              <Text style={styles.amenityText}>+{property.amenities.length - 3} more</Text>
            </View>
          )}
        </View>
        
        <View style={styles.footerRow}>
          <Text style={styles.availableText}>
            Available: {new Date(property.availableFrom).toLocaleDateString()}
          </Text>
          
          {property.utilitiesIncluded && (
            <View style={styles.utilitiesTag}>
              <Text style={styles.utilitiesText}>Utilities Included</Text>
            </View>
          )}
        </View>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: COLORS.surface,
    borderRadius: SIZES.radius,
    overflow: 'hidden',
    marginBottom: SIZES.padding,
  },
  verticalContainer: {
    width: CARD_WIDTH,
  },
  horizontalContainer: {
    flexDirection: 'row',
    width: '100%',
    height: 120,
  },
  verticalImage: {
    width: '100%',
    height: 150,
  },
  horizontalImage: {
    width: 120,
    height: '100%',
  },
  contentContainer: {
    padding: SIZES.base,
    flex: 1,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: SIZES.base / 2,
  },
  priceText: {
    fontSize: SIZES.large,
    fontWeight: 'bold',
    color: COLORS.primary,
  },
  statusContainer: {
    flexDirection: 'row',
  },
  statusBadge: {
    paddingHorizontal: SIZES.base,
    paddingVertical: 2,
    borderRadius: SIZES.radius,
    backgroundColor: COLORS.info,
  },
  activeBadge: {
    backgroundColor: COLORS.success,
  },
  pendingBadge: {
    backgroundColor: COLORS.warning,
  },
  rentedBadge: {
    backgroundColor: COLORS.info,
  },
  expiredBadge: {
    backgroundColor: COLORS.disabled,
  },
  statusText: {
    color: COLORS.surface,
    fontSize: 10,
    fontWeight: 'bold',
  },
  titleText: {
    fontSize: SIZES.medium,
    fontWeight: '600',
    color: COLORS.text,
    marginBottom: SIZES.base / 2,
  },
  addressRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: SIZES.base / 2,
  },
  addressText: {
    fontSize: SIZES.small,
    color: COLORS.textSecondary,
    marginLeft: 2,
    flex: 1,
  },
  featuresRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: SIZES.base / 2,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: SIZES.base,
    marginBottom: SIZES.base / 2,
  },
  featureText: {
    fontSize: SIZES.small,
    color: COLORS.textSecondary,
    marginLeft: 2,
  },
  highlightText: {
    color: COLORS.primary,
    fontWeight: '500',
  },
  amenitiesContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: SIZES.base / 2,
  },
  amenityTag: {
    backgroundColor: COLORS.background,
    borderRadius: SIZES.radius,
    paddingHorizontal: SIZES.base / 2,
    paddingVertical: 2,
    marginRight: SIZES.base / 2,
    marginBottom: SIZES.base / 2,
  },
  amenityText: {
    fontSize: 10,
    color: COLORS.textSecondary,
  },
  footerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: SIZES.base / 2,
  },
  availableText: {
    fontSize: SIZES.small,
    color: COLORS.text,
  },
  utilitiesTag: {
    backgroundColor: COLORS.accent,
    borderRadius: SIZES.radius,
    paddingHorizontal: SIZES.base / 2,
    paddingVertical: 2,
  },
  utilitiesText: {
    fontSize: 10,
    color: COLORS.text,
    fontWeight: '500',
  },
});

export default PropertyCard; 