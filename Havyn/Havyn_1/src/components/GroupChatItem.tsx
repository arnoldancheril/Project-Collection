import React from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { GroupChat } from '../models/Group';

interface GroupChatItemProps {
  group: GroupChat;
  onPress: (group: GroupChat) => void;
}

const GroupChatItem: React.FC<GroupChatItemProps> = ({ group, onPress }) => {
  // Format the time (e.g. "10m ago", "2h ago", "Yesterday")
  const formatTime = (timestamp: any): string => {
    if (!timestamp) return '';
    
    const now = new Date();
    const messageTime = timestamp.toDate();
    const diffMs = now.getTime() - messageTime.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffMins < 60) {
      return `${diffMins}m ago`;
    } else if (diffHours < 24) {
      return `${diffHours}h ago`;
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else {
      return `${diffDays}d ago`;
    }
  };
  
  // Check if there are unread messages for current user
  const hasUnread = group.hasUnreadMessages?.['00001']; // Using hardcoded test user ID

  // Get the last message time or creation time
  const lastActivity = group.lastMessageTimestamp || group.createdAt;
  const timeAgo = formatTime(lastActivity);
  
  return (
    <TouchableOpacity 
      style={styles.container}
      onPress={() => onPress(group)}
      activeOpacity={0.7}
    >
      <Image 
        source={{ uri: group.groupImageUrl }} 
        style={styles.groupImage} 
        defaultSource={{ uri: 'https://via.placeholder.com/100/4a90e2/ffffff?text=Group' }}
      />
      
      <View style={styles.content}>
        <View style={styles.headerRow}>
          <Text style={styles.groupName}>{group.name}</Text>
          <Text style={styles.timeAgo}>{timeAgo}</Text>
        </View>
        
        <Text style={styles.memberCount}>
          {group.memberCount} member{group.memberCount !== 1 ? 's' : ''}
        </Text>
        
        {group.lastMessageText ? (
          <View style={styles.messageRow}>
            {group.lastMessageSender && (
              <Text style={styles.messageSender}>{group.lastMessageSender.split(' ')[0]}: </Text>
            )}
            <Text 
              style={[styles.messagePreview, hasUnread && styles.unreadMessage]} 
              numberOfLines={1}
            >
              {group.lastMessageText}
            </Text>
            {hasUnread && <View style={styles.unreadIndicator} />}
          </View>
        ) : (
          <Text style={styles.description} numberOfLines={1}>
            {group.description}
          </Text>
        )}
      </View>
      
      <Ionicons name="chevron-forward" size={24} color="#999" />
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  groupImage: {
    width: 60,
    height: 60,
    borderRadius: 30,
    marginRight: 16,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  groupName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  timeAgo: {
    fontSize: 13,
    color: '#999',
  },
  memberCount: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  messageRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  messageSender: {
    fontSize: 14,
    fontWeight: '600',
    color: '#555',
  },
  messagePreview: {
    fontSize: 14,
    color: '#666',
    flex: 1,
  },
  unreadMessage: {
    fontWeight: '600',
    color: '#333',
  },
  description: {
    fontSize: 14,
    color: '#666',
    fontStyle: 'italic',
  },
  unreadIndicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#3498db',
    marginLeft: 8,
  },
});

export default GroupChatItem; 