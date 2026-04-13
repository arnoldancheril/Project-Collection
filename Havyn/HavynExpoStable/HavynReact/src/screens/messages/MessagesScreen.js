import React from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  FlatList, 
  Image, 
  TouchableOpacity,
  SafeAreaView
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../../utils/theme';
import { SAMPLE_PROFILES } from '../../utils/sampleProfiles';

// Sample chat previews based on sample profiles
const CHAT_PREVIEWS = SAMPLE_PROFILES.slice(0, 5).map(profile => ({
  id: profile.id,
  name: `${profile.firstName} ${profile.lastName}`,
  message: profile.id === '1' ? 'Hey, I saw your listing and I\'m interested!' : 
           profile.id === '2' ? 'When can I come see the place?' :
           profile.id === '3' ? 'Do you allow pets in the apartment?' :
           profile.id === '4' ? 'Thanks for the information!' :
           'Are utilities included in the rent?',
  timestamp: profile.id === '1' ? 'Just now' : 
             profile.id === '2' ? '10 min ago' :
             profile.id === '3' ? '1 hour ago' :
             profile.id === '4' ? 'Yesterday' :
             '2 days ago',
  unread: profile.id === '1' || profile.id === '3',
  avatar: profile.images[0]
}));

const MessagesScreen = ({ navigation }) => {
  const renderChatItem = ({ item }) => (
    <TouchableOpacity 
      style={styles.chatItem}
      onPress={() => navigation.navigate('ChatScreen', { chat: item })}
    >
      <View style={styles.avatarContainer}>
        <Image source={item.avatar} style={styles.avatar} />
        {item.unread && <View style={styles.unreadBadge} />}
      </View>
      <View style={styles.chatInfo}>
        <View style={styles.chatHeader}>
          <Text style={styles.chatName}>{item.name}</Text>
          <Text style={styles.chatTime}>{item.timestamp}</Text>
        </View>
        <Text 
          style={[styles.chatMessage, item.unread && styles.unreadMessage]}
          numberOfLines={1}
        >
          {item.message}
        </Text>
      </View>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Messages</Text>
        <TouchableOpacity style={styles.filterButton}>
          <Ionicons name="options-outline" size={24} color={COLORS.primary} />
        </TouchableOpacity>
      </View>
      
      {CHAT_PREVIEWS.length > 0 ? (
        <FlatList
          data={CHAT_PREVIEWS}
          renderItem={renderChatItem}
          keyExtractor={item => item.id}
          contentContainerStyle={styles.chatList}
          showsVerticalScrollIndicator={false}
        />
      ) : (
        <View style={styles.emptyContainer}>
          <Ionicons name="chatbubble-ellipses-outline" size={80} color={COLORS.lightGrey} />
          <Text style={styles.emptyTitle}>No messages yet</Text>
          <Text style={styles.emptySubtitle}>
            Your messages with potential roommates will appear here
          </Text>
        </View>
      )}
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
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: COLORS.text,
  },
  filterButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#f5f5f5',
    justifyContent: 'center',
    alignItems: 'center',
  },
  chatList: {
    paddingHorizontal: 20,
  },
  chatItem: {
    flexDirection: 'row',
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#f5f5f5',
  },
  avatarContainer: {
    position: 'relative',
    marginRight: 15,
  },
  avatar: {
    width: 60,
    height: 60,
    borderRadius: 30,
  },
  unreadBadge: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 14,
    height: 14,
    borderRadius: 7,
    backgroundColor: COLORS.primary,
    borderWidth: 2,
    borderColor: '#fff',
  },
  chatInfo: {
    flex: 1,
    justifyContent: 'center',
  },
  chatHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  chatName: {
    fontSize: 16,
    fontWeight: '600',
    color: COLORS.text,
  },
  chatTime: {
    fontSize: 12,
    color: COLORS.textSecondary,
  },
  chatMessage: {
    fontSize: 14,
    color: COLORS.textSecondary,
  },
  unreadMessage: {
    fontWeight: '600',
    color: COLORS.text,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 30,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: COLORS.text,
    marginTop: 20,
    marginBottom: 10,
  },
  emptySubtitle: {
    fontSize: 16,
    color: COLORS.textSecondary,
    textAlign: 'center',
    lineHeight: 22,
  },
});

export default MessagesScreen; 