import React, { useState, useEffect } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  FlatList, 
  Image, 
  TouchableOpacity, 
  Modal,
  SafeAreaView,
  ScrollView,
  ActivityIndicator
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { User } from '../../src/models/User';
import { Match } from '../../src/models/Match';
import { DetailedProfileView } from '../../src/components';
import { generateRandomMatches, getMatchedUsers, sortMatchesByRecency } from '../../src/services/matchService';

export default function MatchesScreen() {
  const [matches, setMatches] = useState<Match[]>([]);
  const [matchedUsers, setMatchedUsers] = useState<User[]>([]);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [showDetailedView, setShowDetailedView] = useState(false);
  const [showChatModal, setShowChatModal] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadMatches();
  }, []);

  const loadMatches = async () => {
    try {
      setLoading(true);
      
      // Generate random matches for testing
      const randomMatchCount = Math.floor(Math.random() * 6) + 5; // 5-10 matches
      const generatedMatches = await generateRandomMatches(randomMatchCount);
      
      // Sort matches by recency
      const sortedMatches = sortMatchesByRecency(generatedMatches);
      setMatches(sortedMatches);
      
      // Get the user objects for these matches
      const users = await getMatchedUsers(sortedMatches);
      setMatchedUsers(users);
      
      setLoading(false);
    } catch (error) {
      console.error('Error loading matches:', error);
      setLoading(false);
    }
  };

  const handleOpenUserDetails = (user: User) => {
    setSelectedUser(user);
    setShowDetailedView(true);
  };

  const handleOpenChat = (user: User) => {
    setSelectedUser(user);
    setShowChatModal(true);
  };

  const renderMatchItem = ({ item }: { item: User }) => {
    // Find the match object for this user
    const matchObj = matches.find(m => 
      m.recipientId === (item.userId || item.id) || m.initiatorId === (item.userId || item.id)
    );
    
    // Get user profile image
    const profileImage = item.images?.profile?.[0] || item.profileImageUrl || 'https://via.placeholder.com/100';
    
    // Check if there are unread messages
    const hasUnread = matchObj?.hasUnreadMessages;
    
    return (
      <TouchableOpacity 
        style={styles.matchItem}
        onPress={() => handleOpenUserDetails(item)}
      >
        <View style={styles.avatarContainer}>
          <Image source={{ uri: profileImage }} style={styles.avatar} />
          {hasUnread && <View style={styles.unreadIndicator} />}
        </View>
        
        <View style={styles.userInfo}>
          <Text style={styles.userName}>{item.name}</Text>
          <Text style={styles.userDetails}>{item.age}, Chicago</Text>
        </View>
        
        <TouchableOpacity 
          style={styles.messageButton}
          onPress={() => handleOpenChat(item)}
        >
          <Ionicons name="chevron-forward" size={24} color="#999" />
        </TouchableOpacity>
      </TouchableOpacity>
    );
  };

  // Simple chat modal (placeholder for actual chat interface)
  const ChatModal = () => {
    if (!selectedUser) return null;
    
    return (
      <Modal
        animationType="slide"
        transparent={false}
        visible={showChatModal}
        onRequestClose={() => setShowChatModal(false)}
      >
        <SafeAreaView style={styles.chatContainer}>
          <View style={styles.chatHeader}>
            <TouchableOpacity onPress={() => setShowChatModal(false)}>
              <Ionicons name="arrow-back" size={24} color="#3498db" />
            </TouchableOpacity>
            <Text style={styles.chatHeaderTitle}>Chat with {selectedUser.name}</Text>
            <View style={{width: 24}} />
          </View>
          
          <ScrollView style={styles.chatMessages}>
            <View style={styles.emptyChat}>
              <Ionicons name="chatbubbles-outline" size={60} color="#ccc" />
              <Text style={styles.emptyChatText}>No messages yet</Text>
              <Text style={styles.emptyChatSubtext}>Say hello to {selectedUser.name}!</Text>
            </View>
          </ScrollView>
          
          <View style={styles.chatInputContainer}>
            <View style={styles.chatInput}>
              <Text style={styles.chatInputPlaceholder}>Type a message...</Text>
            </View>
            <TouchableOpacity style={styles.sendButton}>
              <Ionicons name="send" size={24} color="#3498db" />
            </TouchableOpacity>
          </View>
        </SafeAreaView>
      </Modal>
    );
  };

  if (loading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#3498db" />
        <Text style={styles.loadingText}>Loading matches...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Matches</Text>
      </View>
      
      {matchedUsers.length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="heart-outline" size={80} color="#ccc" />
          <Text style={styles.emptyStateTitle}>No matches yet</Text>
          <Text style={styles.emptyStateSubtitle}>
            Swipe right on profiles you're interested in to find potential roommates
          </Text>
        </View>
      ) : (
        <FlatList
          data={matchedUsers}
          keyExtractor={(item) => item.userId || item.id}
          renderItem={renderMatchItem}
          contentContainerStyle={styles.list}
        />
      )}
      
      {/* Detailed Profile View Modal */}
      <DetailedProfileView
        user={selectedUser}
        visible={showDetailedView}
        onClose={() => setShowDetailedView(false)}
      />
      
      {/* Chat Modal */}
      <ChatModal />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#666',
  },
  header: {
    padding: 20,
    paddingTop: 40,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
  },
  list: {
    paddingVertical: 10,
  },
  matchItem: {
    backgroundColor: 'white',
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  avatarContainer: {
    position: 'relative',
  },
  avatar: {
    width: 60,
    height: 60,
    borderRadius: 30,
    marginRight: 16,
  },
  unreadIndicator: {
    position: 'absolute',
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#3498db',
    borderWidth: 2,
    borderColor: 'white',
    top: 0,
    right: 16,
  },
  userInfo: {
    flex: 1,
  },
  userName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  userDetails: {
    fontSize: 14,
    color: '#666',
  },
  messageButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  emptyStateTitle: {
    fontSize: 22,
    fontWeight: '600',
    color: '#333',
    marginTop: 16,
    marginBottom: 8,
  },
  emptyStateSubtitle: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    lineHeight: 24,
  },
  chatContainer: {
    flex: 1,
    backgroundColor: 'white',
  },
  chatHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  chatHeaderTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  chatMessages: {
    flex: 1,
  },
  emptyChat: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 100,
  },
  emptyChatText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginTop: 16,
    marginBottom: 8,
  },
  emptyChatSubtext: {
    fontSize: 16,
    color: '#666',
  },
  chatInputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
  },
  chatInput: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 12,
    marginRight: 12,
  },
  chatInputPlaceholder: {
    color: '#999',
  },
  sendButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#f5f5f5',
    justifyContent: 'center',
    alignItems: 'center',
  },
}); 