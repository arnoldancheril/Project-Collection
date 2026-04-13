import React, { useState, useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Alert, RefreshControl, ScrollView, TouchableOpacity } from 'react-native';
import { ProfileCard, FilterBar, DetailedProfileView } from '../../src/components';
import { User } from '../../src/models/User';
import { getAllUsers } from '../../src/services/sampleDataService';
import { simulateMatchDecision } from '../../src/services/matchService';
import { Ionicons } from '@expo/vector-icons';

export default function HomeScreen() {
  const [allUsers, setAllUsers] = useState<User[]>([]);
  const [filteredUsers, setFilteredUsers] = useState<User[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [connectionTest, setConnectionTest] = useState('Testing Firebase connection...');
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [showDetailedView, setShowDetailedView] = useState(false);

  
  // Reference to track if we're in the process of changing profiles
  const isChangingProfile = useRef(false);

  const testFirebaseConnection = async () => {
    try {
      console.log('Testing Firebase connection...');
      await getAllUsers();
      setConnectionTest('✅ Firebase connected successfully');
    } catch (error) {
      console.error('Firebase connection error:', error);
      setConnectionTest(`❌ Firebase connection failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const loadUsers = async () => {
    try {
      const fetchedUsers = await getAllUsers();
      setAllUsers(fetchedUsers);
      setFilteredUsers(fetchedUsers);
      setCurrentIndex(0); // Reset to first user
      setConnectionTest('✅ Firebase connected successfully');
      
      // Log image data for debugging
      if (fetchedUsers.length > 0) {
        const currentUser = fetchedUsers[0];
        console.log('First user image data:');
        console.log('- Legacy profileImageUrl:', currentUser.profileImageUrl);
        console.log('- New images.profile:', currentUser.images?.profile);
        console.log('- Property images:', currentUser.images?.property);
      }
    } catch (error) {
      console.error('Error loading users:', error);
      setConnectionTest(`❌ Error loading users: ${error instanceof Error ? error.message : 'Unknown error'}`);
      Alert.alert('Error', 'Failed to load user profiles');
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadUsers();
    setRefreshing(false);
  };

  const handleSwipeLeft = (user: User) => {
    console.log('Swiped LEFT (Pass) on:', user.name);
    
    // Immediately move to next user without waiting for animation
    if (currentIndex < filteredUsers.length - 1) {
      setCurrentIndex(currentIndex + 1);
    } else {
      // No more users
      setCurrentIndex(filteredUsers.length);
    }
  };

  const handleSwipeRight = (user: User) => {
    console.log('Swiped RIGHT (Like) on:', user.name);
    
    // Use the match service to simulate a match decision
    const isMatch = simulateMatchDecision(0.7); // 70% match rate
    
    if (isMatch) {
      console.log('🎉 New match with:', user.name);
      // In a real app, we would store this match in Firestore
      // For now, just show a notification
      Alert.alert(
        'New Match!', 
        `You matched with ${user.name}! Check your matches tab.`,
        [{ text: 'OK', onPress: () => console.log('Match alert closed') }]
      );
    }
    
    // Immediately move to next user without waiting for animation
    if (currentIndex < filteredUsers.length - 1) {
      setCurrentIndex(currentIndex + 1);
    } else {
      // No more users
      setCurrentIndex(filteredUsers.length);
    }
  };

  const handleMoreInfo = (user: User) => {
    setSelectedUser(user);
    setShowDetailedView(true);
  };

  const handleDetailedLike = (user: User) => {
    setShowDetailedView(false);
    handleSwipeRight(user);
  };

  const handleDetailedPass = (user: User) => {
    setShowDetailedView(false);
    handleSwipeLeft(user);
  };

  const handleFilterChange = (_filters: any, newFilteredUsers: User[]) => {
    setFilteredUsers(newFilteredUsers);
    setCurrentIndex(0); // Reset to first filtered user
  };



  useEffect(() => {
    testFirebaseConnection();
    loadUsers();
  }, []);

  if (loading && allUsers.length === 0) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.loadingText}>Loading profiles...</Text>
        <Text style={styles.connectionText}>{connectionTest}</Text>
      </View>
    );
  }

  const currentUser = filteredUsers[currentIndex];
  const hasMoreUsers = currentIndex < filteredUsers.length;
  
  // Check if current user has multiple images
  const hasMultipleImages = currentUser?.images?.profile && currentUser.images.profile.length > 1;

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <View style={styles.headerContent}>
          <View>
            <Text style={styles.title}>Discover</Text>
            <Text style={styles.subtitle}>
              {hasMoreUsers 
                ? `${currentIndex + 1} of ${filteredUsers.length} profiles` 
                : 'No more profiles'}
            </Text>
          </View>
        </View>
      </View>

      <FilterBar 
        allUsers={allUsers} 
        onFilterChange={handleFilterChange}
      />

      <ScrollView 
        style={styles.scrollContainer}
        contentContainerStyle={styles.scrollContent}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        showsVerticalScrollIndicator={false}
      >
        {!hasMoreUsers ? (
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyText}>🎉 You've seen all profiles!</Text>
            <Text style={styles.emptySubtext}>
              Pull down to refresh and see if there are new matches
            </Text>
          </View>
        ) : (
          <View style={styles.cardContainer}>
            <ProfileCard 
              user={currentUser}
              onSwipeLeft={handleSwipeLeft}
              onSwipeRight={handleSwipeRight}
              onMoreInfo={handleMoreInfo}
            />
          </View>
        )}
      </ScrollView>

      {/* Detailed Profile View Modal */}
      <DetailedProfileView
        user={selectedUser}
        visible={showDetailedView}
        onClose={() => setShowDetailedView(false)}
        onLike={handleDetailedLike}
        onPass={handleDetailedPass}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  header: {
    padding: 20,
    paddingTop: 40,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'flex-start',
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
  },
  scrollContainer: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    paddingVertical: 10,
    paddingBottom: 30,
  },
  cardContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingBottom: 20,
  },
  loadingText: {
    fontSize: 18,
    color: '#666',
    marginBottom: 10,
  },
  connectionText: {
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  emptyText: {
    fontSize: 24,
    color: '#333',
    marginBottom: 12,
    textAlign: 'center',
    fontWeight: '600',
  },
  emptySubtext: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    lineHeight: 24,
  },
}); 