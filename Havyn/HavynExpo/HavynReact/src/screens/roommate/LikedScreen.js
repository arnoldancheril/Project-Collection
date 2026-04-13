import React from 'react';
import { View, Text, StyleSheet, FlatList, Image, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../../utils/theme';
import { useProfile } from '../../contexts/ProfileContext';

const LikedScreen = ({ navigation }) => {
  const { likedProfiles, removeLiked } = useProfile();

  // Handle navigation to profile details with serialized data
  const navigateToProfile = (profile) => {
    // Create a serializable copy and convert dates to strings
    const serializedProfile = {...profile};
    if (serializedProfile.moveInDate && serializedProfile.moveInDate instanceof Date) {
      serializedProfile.moveInDate = serializedProfile.moveInDate.toISOString();
    }
    
    navigation.navigate('DetailedProfile', { profile: serializedProfile });
  };

  const renderItem = ({ item }) => (
    <TouchableOpacity 
      style={styles.profileCard}
      onPress={() => navigateToProfile(item)}
    >
      <Image 
        source={item.images && item.images.length > 0 ? item.images[0] : require('../../../assets/person-placeholder.jpg')} 
        style={styles.profileImage} 
      />
      <View style={styles.profileInfo}>
        <Text style={styles.profileName}>
          {item.firstName || item.name}
          {item.age ? `, ${item.age}` : ''}
        </Text>
        <Text style={styles.profileOccupation}>{item.occupation || 'Not specified'}</Text>
        <Text style={styles.profileDistance}>
          {item.location?.neighborhood || item.location?.city || 'Unknown'}
          {item.location?.state ? `, ${item.location.state}` : ''}
        </Text>
        <View style={styles.compatibilityContainer}>
          <Ionicons name="heart" size={16} color={COLORS.primary} />
          <Text style={styles.compatibilityText}>Liked Profile</Text>
        </View>
      </View>
      <TouchableOpacity 
        style={styles.actionButton}
        onPress={() => removeLiked(item.id)}
      >
        <Ionicons name="close" size={20} color="#fff" />
      </TouchableOpacity>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <FlatList
        data={likedProfiles}
        renderItem={renderItem}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.listContainer}
        showsVerticalScrollIndicator={false}
        ListHeaderComponent={
          <View style={styles.header}>
            <Text style={styles.headerTitle}>Profiles You've Liked</Text>
            <Text style={styles.headerSubtitle}>You've liked {likedProfiles.length} profiles so far</Text>
          </View>
        }
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Ionicons name="heart-outline" size={60} color={COLORS.primary} />
            <Text style={styles.emptyText}>You haven't liked any profiles yet</Text>
            <Text style={styles.emptySubtext}>Start swiping to find your perfect roommate match!</Text>
          </View>
        }
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f8f8',
  },
  listContainer: {
    padding: 16,
    paddingBottom: 100, // Add extra padding at the bottom for better scrolling
  },
  header: {
    marginBottom: 20,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#666',
  },
  profileCard: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 12,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  profileImage: {
    width: 80,
    height: 80,
    borderRadius: 8,
    marginRight: 12,
  },
  profileInfo: {
    flex: 1,
    justifyContent: 'center',
  },
  profileName: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  profileOccupation: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  profileDistance: {
    fontSize: 12,
    color: '#888',
    marginBottom: 6,
  },
  compatibilityContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  compatibilityText: {
    marginLeft: 5,
    fontSize: 14,
    color: COLORS.primary,
    fontWeight: '500',
  },
  actionButton: {
    backgroundColor: COLORS.secondary,
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    alignSelf: 'center',
  },
  emptyContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 60,
    padding: 20,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: 20,
    marginBottom: 10,
    color: '#555',
  },
  emptySubtext: {
    fontSize: 14,
    color: '#888',
    textAlign: 'center',
    paddingHorizontal: 20,
  },
});

export default LikedScreen; 