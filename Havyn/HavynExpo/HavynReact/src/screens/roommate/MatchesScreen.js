import React, { useState } from 'react';
import { View, Text, StyleSheet, FlatList, Image, TouchableOpacity, Dimensions } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../../utils/theme';
import { useProfile } from '../../contexts/ProfileContext';

const SAMPLE_GROUP_CHATS = [
  {
    id: '1',
    name: 'Chicago Downtown Apartments',
    members: 8,
    lastMessage: 'Has anyone toured the units on Michigan Ave?',
    time: '10m ago',
  },
  {
    id: '2',
    name: 'Budget Friendly Options',
    members: 5,
    lastMessage: 'I found a great deal in Wicker Park',
    time: '2h ago',
  },
  {
    id: '3',
    name: 'Lincoln Park Roommates',
    members: 6,
    lastMessage: 'Looking for 2 more people to join our apartment',
    time: 'Yesterday',
  },
];

const { width } = Dimensions.get('window');

const MatchesScreen = ({ navigation }) => {
  const [activeTab, setActiveTab] = useState('matches'); // 'matches' or 'groupChats'
  const { matchedProfiles } = useProfile();

  const renderMatchItem = ({ item }) => (
    <TouchableOpacity
      style={styles.matchItem}
      onPress={() => navigation.navigate('DetailedProfile', { profile: item })}
    >
      <View style={styles.avatarContainer}>
        <Image source={item.images[0]} style={styles.avatar} />
        <View style={styles.onlineIndicator} />
      </View>
      
      <View style={styles.matchInfo}>
        <Text style={styles.matchName}>{item.firstName || item.name}, {item.age}</Text>
        <Text style={styles.matchLocation}>{item.location?.city || 'Chicago'}</Text>
      </View>
      
      <TouchableOpacity 
        style={styles.messageButton}
        onPress={() => navigation.navigate('ChatScreen', { profile: item })}
      >
        <Text style={styles.messageButtonText}>Message</Text>
      </TouchableOpacity>
    </TouchableOpacity>
  );

  const renderGroupChatItem = ({ item }) => (
    <TouchableOpacity 
      style={styles.groupChatItem}
      onPress={() => navigation.navigate('GroupChatDetail', { group: item })}
    >
      <View style={styles.groupChatContent}>
        <Text style={styles.groupChatName}>{item.name}</Text>
        <Text style={styles.groupChatMembers}>{item.members} members</Text>
        <Text style={styles.groupChatMessage} numberOfLines={1}>{item.lastMessage}</Text>
      </View>
      <Text style={styles.groupChatTime}>{item.time}</Text>
    </TouchableOpacity>
  );

  const renderCreateGroupChat = () => (
    <TouchableOpacity 
      style={styles.createGroupButton}
      onPress={() => navigation.navigate('CreateGroupChat')}
    >
      <Ionicons name="add" size={24} color="#fff" />
      <Text style={styles.createGroupText}>Create New Group Chat</Text>
    </TouchableOpacity>
  );

  const renderGroupChatsInfo = () => (
    <View style={styles.groupInfoContainer}>
      <Text style={styles.groupInfoTitle}>Why Join Group Chats?</Text>
      <View style={styles.groupInfoItem}>
        <Ionicons name="business-outline" size={24} color={COLORS.primary} style={styles.groupInfoIcon} />
        <Text style={styles.groupInfoText}>Discuss Properties</Text>
      </View>
      <Text style={styles.groupInfoSubtext}>
        Share insights about different apartments and neighborhoods
      </Text>
    </View>
  );

  return (
    <View style={styles.container}>
      {/* Tab Navigation */}
      <View style={styles.tabContainer}>
        <TouchableOpacity 
          style={[styles.tab, activeTab === 'matches' && styles.activeTab]}
          onPress={() => setActiveTab('matches')}
        >
          <Text style={[styles.tabText, activeTab === 'matches' && styles.activeTabText]}>Matches</Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[styles.tab, activeTab === 'groupChats' && styles.activeTab]}
          onPress={() => setActiveTab('groupChats')}
        >
          <Text style={[styles.tabText, activeTab === 'groupChats' && styles.activeTabText]}>Group Chats</Text>
        </TouchableOpacity>
      </View>
      
      {/* Matches Tab */}
      {activeTab === 'matches' && (
        <FlatList
          data={matchedProfiles}
          renderItem={renderMatchItem}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.listContainer}
          showsVerticalScrollIndicator={false}
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <Ionicons name="people-outline" size={60} color={COLORS.primary} />
              <Text style={styles.emptyText}>No matches yet</Text>
              <Text style={styles.emptySubtext}>
                When you match with other users, you'll be able to message them here.
              </Text>
            </View>
          }
        />
      )}
      
      {/* Group Chats Tab */}
      {activeTab === 'groupChats' && (
        <FlatList
          data={SAMPLE_GROUP_CHATS}
          renderItem={renderGroupChatItem}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.listContainer}
          showsVerticalScrollIndicator={false}
          ListHeaderComponent={renderCreateGroupChat}
          ListFooterComponent={renderGroupChatsInfo}
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <Ionicons name="chatbubbles-outline" size={60} color={COLORS.primary} />
              <Text style={styles.emptyText}>No group chats yet</Text>
              <Text style={styles.emptySubtext}>
                Create a group chat to discuss properties with others.
              </Text>
            </View>
          }
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  tabContainer: {
    flexDirection: 'row',
    borderBottomWidth: 1,
    borderBottomColor: '#EEEEEE',
  },
  tab: {
    flex: 1,
    paddingVertical: 15,
    alignItems: 'center',
  },
  activeTab: {
    borderBottomWidth: 2,
    borderBottomColor: COLORS.primary,
  },
  tabText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#757575',
  },
  activeTabText: {
    color: COLORS.primary,
    fontWeight: '600',
  },
  listContainer: {
    paddingHorizontal: 16,
    paddingTop: 10,
    paddingBottom: 20,
  },
  matchItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 15,
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
  },
  onlineIndicator: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 14,
    height: 14,
    borderRadius: 7,
    backgroundColor: '#4CAF50',
    borderWidth: 2,
    borderColor: '#fff',
  },
  matchInfo: {
    flex: 1,
    marginLeft: 15,
  },
  matchName: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 3,
  },
  matchLocation: {
    fontSize: 14,
    color: '#757575',
  },
  messageButton: {
    backgroundColor: COLORS.primary,
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
  },
  messageButtonText: {
    color: '#fff',
    fontWeight: '500',
    fontSize: 14,
  },
  groupChatItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  groupChatContent: {
    flex: 1,
  },
  groupChatName: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  groupChatMembers: {
    fontSize: 12,
    color: '#757575',
    marginBottom: 4,
  },
  groupChatMessage: {
    fontSize: 14,
    color: '#424242',
  },
  groupChatTime: {
    fontSize: 12,
    color: '#9E9E9E',
    marginLeft: 15,
  },
  createGroupButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.primary,
    padding: 15,
    borderRadius: 12,
    marginBottom: 20,
  },
  createGroupText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 16,
    marginLeft: 12,
  },
  groupInfoContainer: {
    marginTop: 30,
    backgroundColor: '#f8f8f8',
    padding: 20,
    borderRadius: 12,
  },
  groupInfoTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  groupInfoItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  groupInfoIcon: {
    marginRight: 10,
  },
  groupInfoText: {
    fontSize: 16,
    fontWeight: '600',
  },
  groupInfoSubtext: {
    fontSize: 14,
    color: '#757575',
    marginLeft: 34,
  },
  emptyContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 50,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: 15,
    marginBottom: 10,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#757575',
    textAlign: 'center',
    marginHorizontal: 30,
  },
});

export default MatchesScreen; 