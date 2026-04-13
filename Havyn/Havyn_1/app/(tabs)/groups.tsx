import React, { useState, useEffect } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  TouchableOpacity, 
  FlatList, 
  ActivityIndicator,
  SectionList
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { 
  GroupChatItem, 
  CreateGroupModal, 
  GroupChatView, 
  GroupMembersModal, 
  DetailedProfileView 
} from '../../src/components';
import { 
  generateSampleGroups, 
  getSuggestedGroups, 
  sortGroupsByActivity, 
  joinGroup, 
  createNewGroup 
} from '../../src/services/groupService';
import { GroupChat } from '../../src/models/Group';
import { User } from '../../src/models/User';

export default function GroupsScreen() {
  const [yourGroups, setYourGroups] = useState<GroupChat[]>([]);
  const [suggestedGroups, setSuggestedGroups] = useState<GroupChat[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedGroup, setSelectedGroup] = useState<GroupChat | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showChatView, setShowChatView] = useState(false);
  const [showMembersModal, setShowMembersModal] = useState(false);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [showUserProfile, setShowUserProfile] = useState(false);

  useEffect(() => {
    loadGroups();
  }, []);

  const loadGroups = async () => {
    try {
      setLoading(true);
      // Generate 3 groups you're a member of
      const groups = await generateSampleGroups(3);
      
      // Sort by activity
      const sortedGroups = sortGroupsByActivity(groups);
      setYourGroups(sortedGroups);
      
      // Get 2 suggested groups
      const suggested = await getSuggestedGroups(await generateSampleGroups(5));
      setSuggestedGroups(suggested.slice(0, 2));
    } catch (error) {
      console.error('Error loading groups:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateGroup = async (
    name: string, 
    description: string, 
    isPublic: boolean, 
    tags: string[]
  ) => {
    try {
      const newGroup = await createNewGroup(name, description, [], isPublic, tags);
      setYourGroups([newGroup, ...yourGroups]);
    } catch (error) {
      console.error('Error creating group:', error);
    }
  };

  const handleJoinGroup = async (group: GroupChat) => {
    try {
      // Add current user to group members
      const updatedGroups = await joinGroup(group.id, [...yourGroups, ...suggestedGroups]);
      
      // Find the updated group
      const joinedGroup = updatedGroups.find(g => g.id === group.id);
      
      if (joinedGroup) {
        // Remove from suggested groups
        setSuggestedGroups(suggestedGroups.filter(g => g.id !== group.id));
        
        // Add to your groups
        setYourGroups([joinedGroup, ...yourGroups]);
      }
    } catch (error) {
      console.error('Error joining group:', error);
    }
  };

  const handleOpenGroup = (group: GroupChat) => {
    setSelectedGroup(group);
    setShowChatView(true);
  };

  const handleViewMembers = (group: GroupChat) => {
    setSelectedGroup(group);
    setShowMembersModal(true);
  };

  const handleViewUserProfile = (user: User) => {
    setSelectedUser(user);
    setShowUserProfile(true);
  };

  const renderSectionHeader = ({ section }: { section: { title: string } }) => (
    <View style={styles.sectionHeader}>
      <Text style={styles.sectionTitle}>{section.title}</Text>
    </View>
  );

  const renderGroup = ({ item }: { item: GroupChat }) => (
    <GroupChatItem 
      group={item} 
      onPress={handleOpenGroup} 
    />
  );

  const renderSuggestedGroup = ({ item }: { item: GroupChat }) => (
    <View style={styles.suggestedGroup}>
      <GroupChatItem 
        group={item} 
        onPress={handleOpenGroup} 
      />
      <TouchableOpacity 
        style={styles.joinButton}
        onPress={() => handleJoinGroup(item)}
      >
        <Text style={styles.joinButtonText}>Join Group</Text>
      </TouchableOpacity>
    </View>
  );

  // Empty state UI for each section
  const renderEmptyYourGroups = () => (
    <View style={styles.emptyContainer}>
      <Text style={styles.emptyText}>You haven't joined any groups yet</Text>
      <Text style={styles.emptySubtext}>
        Create a new group or join a suggested group
      </Text>
    </View>
  );

  const renderEmptySuggested = () => (
    <View style={styles.emptyContainer}>
      <Text style={styles.emptyText}>No suggested groups available</Text>
      <Text style={styles.emptySubtext}>
        Try creating your own group to connect with others
      </Text>
    </View>
  );

  // "Why Join Group Chats?" section
  const renderWhyJoinSection = () => (
    <View style={styles.whyJoinSection}>
      <Text style={styles.whyJoinTitle}>Why Join Group Chats?</Text>
      
      <View style={styles.whyJoinItem}>
        <Ionicons name="home-outline" size={24} color="#3498db" style={styles.whyJoinIcon} />
        <View style={styles.whyJoinContent}>
          <Text style={styles.whyJoinItemTitle}>Discuss Properties</Text>
          <Text style={styles.whyJoinItemText}>
            Share insights about different apartments and neighborhoods
          </Text>
        </View>
      </View>
      
      <View style={styles.whyJoinItem}>
        <Ionicons name="people-outline" size={24} color="#3498db" style={styles.whyJoinIcon} />
        <View style={styles.whyJoinContent}>
          <Text style={styles.whyJoinItemTitle}>Find Roommate Groups</Text>
          <Text style={styles.whyJoinItemText}>
            Connect with multiple potential roommates at once
          </Text>
        </View>
      </View>
      
      <View style={styles.whyJoinItem}>
        <Ionicons name="cash-outline" size={24} color="#3498db" style={styles.whyJoinIcon} />
        <View style={styles.whyJoinContent}>
          <Text style={styles.whyJoinItemTitle}>Split Costs</Text>
          <Text style={styles.whyJoinItemText}>
            Coordinate with multiple roommates on shared expenses
          </Text>
        </View>
      </View>
    </View>
  );

  // Prepare data for section list
  const sections = [
    {
      title: 'Your Group Chats',
      data: yourGroups.length > 0 ? yourGroups : [],
      renderItem: renderGroup,
      ListEmptyComponent: renderEmptyYourGroups,
    },
    {
      title: 'Suggested Group Chats',
      data: suggestedGroups.length > 0 ? suggestedGroups : [],
      renderItem: renderSuggestedGroup,
      ListEmptyComponent: renderEmptySuggested,
    },
  ];

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#3498db" />
        <Text style={styles.loadingText}>Loading group chats...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Group Chats</Text>
        <TouchableOpacity 
          style={styles.createButton}
          onPress={() => setShowCreateModal(true)}
        >
          <Ionicons name="add" size={24} color="white" />
          <Text style={styles.createButtonText}>Create New Group Chat</Text>
        </TouchableOpacity>
      </View>
      
      <SectionList
        sections={sections}
        keyExtractor={(item) => item.id}
        renderSectionHeader={renderSectionHeader}
        ListFooterComponent={renderWhyJoinSection}
        stickySectionHeadersEnabled={false}
      />
      
      {/* Create Group Modal */}
      <CreateGroupModal 
        visible={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onCreateGroup={handleCreateGroup}
      />
      
      {/* Group Chat View */}
      <GroupChatView 
        visible={showChatView}
        group={selectedGroup}
        onClose={() => setShowChatView(false)}
        onViewMembers={handleViewMembers}
      />
      
      {/* Group Members Modal */}
      <GroupMembersModal 
        visible={showMembersModal}
        group={selectedGroup}
        onClose={() => setShowMembersModal(false)}
        onViewUserProfile={handleViewUserProfile}
      />
      
      {/* User Profile Modal */}
      <DetailedProfileView 
        user={selectedUser}
        visible={showUserProfile}
        onClose={() => setShowUserProfile(false)}
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
    marginBottom: 16,
  },
  createButton: {
    backgroundColor: '#3498db',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 25,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  createButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  sectionHeader: {
    backgroundColor: '#f5f5f5',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  emptyContainer: {
    padding: 30,
    alignItems: 'center',
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  emptyText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    textAlign: 'center',
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
  suggestedGroup: {
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  joinButton: {
    backgroundColor: '#3498db',
    marginHorizontal: 16,
    marginBottom: 16,
    paddingVertical: 10,
    borderRadius: 8,
    alignItems: 'center',
  },
  joinButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  whyJoinSection: {
    margin: 16,
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  whyJoinTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 16,
  },
  whyJoinItem: {
    flexDirection: 'row',
    marginBottom: 16,
    alignItems: 'flex-start',
  },
  whyJoinIcon: {
    marginRight: 16,
    marginTop: 2,
  },
  whyJoinContent: {
    flex: 1,
  },
  whyJoinItemTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  whyJoinItemText: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
}); 