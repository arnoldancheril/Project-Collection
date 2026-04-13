import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Modal,
  TouchableOpacity,
  FlatList,
  Image,
  ActivityIndicator,
  SafeAreaView,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { GroupChat } from '../models/Group';
import { User } from '../models/User';
import { getGroupMembers } from '../services/groupService';

interface GroupMembersModalProps {
  visible: boolean;
  group: GroupChat | null;
  onClose: () => void;
  onViewUserProfile?: (user: User) => void;
}

const GroupMembersModal: React.FC<GroupMembersModalProps> = ({
  visible,
  group,
  onClose,
  onViewUserProfile,
}) => {
  const [members, setMembers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (group && visible) {
      loadMembers();
    }
  }, [group, visible]);

  const loadMembers = async () => {
    if (!group) return;
    
    try {
      setLoading(true);
      const groupMembers = await getGroupMembers(group);
      setMembers(groupMembers);
    } catch (error) {
      console.error('Error loading group members:', error);
    } finally {
      setLoading(false);
    }
  };

  const isCurrentUser = (userId: string) => {
    return userId === '00001'; // For testing, replace with auth context
  };

  const isCreator = (userId: string) => {
    return group?.createdBy === userId;
  };

  const renderMemberItem = ({ item }: { item: User }) => {
    const userId = item.userId || item.id;
    const currentUser = isCurrentUser(userId);
    const creator = isCreator(userId);
    
    return (
      <TouchableOpacity 
        style={styles.memberItem}
        onPress={() => onViewUserProfile?.(item)}
        disabled={!onViewUserProfile}
      >
        <Image 
          source={{ uri: item.images?.profile?.[0] || item.profileImageUrl }} 
          style={styles.memberAvatar}
          defaultSource={{ uri: 'https://via.placeholder.com/100/4a90e2/ffffff?text=User' }}
        />
        
        <View style={styles.memberInfo}>
          <Text style={styles.memberName}>
            {item.name}{currentUser ? ' (You)' : ''}
          </Text>
          <Text style={styles.memberDetails}>
            {item.age}, Chicago
          </Text>
        </View>
        
        {creator && (
          <View style={styles.creatorBadge}>
            <Text style={styles.creatorText}>Creator</Text>
          </View>
        )}
      </TouchableOpacity>
    );
  };

  if (!group) return null;

  return (
    <Modal
      visible={visible}
      animationType="slide"
      onRequestClose={onClose}
    >
      <SafeAreaView style={styles.container}>
        <View style={styles.header}>
          <TouchableOpacity onPress={onClose} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color="#3498db" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>
            Group Members ({group.memberCount})
          </Text>
          <View style={styles.rightPlaceholder} />
        </View>
        
        {loading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#3498db" />
            <Text style={styles.loadingText}>Loading members...</Text>
          </View>
        ) : (
          <FlatList
            data={members}
            renderItem={renderMemberItem}
            keyExtractor={(item) => item.userId || item.id}
            contentContainerStyle={styles.membersList}
          />
        )}
      </SafeAreaView>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'white',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  backButton: {
    padding: 4,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  rightPlaceholder: {
    width: 32,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#666',
  },
  membersList: {
    padding: 16,
  },
  memberItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  memberAvatar: {
    width: 50,
    height: 50,
    borderRadius: 25,
    marginRight: 16,
  },
  memberInfo: {
    flex: 1,
  },
  memberName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  memberDetails: {
    fontSize: 14,
    color: '#666',
  },
  creatorBadge: {
    backgroundColor: '#3498db',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  creatorText: {
    fontSize: 12,
    color: 'white',
    fontWeight: '500',
  },
});

export default GroupMembersModal; 