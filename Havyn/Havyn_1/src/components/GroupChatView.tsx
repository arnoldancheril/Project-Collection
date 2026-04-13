import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Modal,
  TouchableOpacity,
  FlatList,
  TextInput,
  Image,
  SafeAreaView,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { GroupChat, GroupMessage } from '../models/Group';
import { generateGroupMessages } from '../services/groupService';
import { User } from '../models/User';

interface GroupChatViewProps {
  visible: boolean;
  group: GroupChat | null;
  onClose: () => void;
  onViewMembers: (group: GroupChat) => void;
}

const GroupChatView: React.FC<GroupChatViewProps> = ({
  visible,
  group,
  onClose,
  onViewMembers,
}) => {
  const [messages, setMessages] = useState<GroupMessage[]>([]);
  const [messageText, setMessageText] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (group && visible) {
      loadMessages();
    }
  }, [group, visible]);

  const loadMessages = async () => {
    if (!group) return;
    
    try {
      setLoading(true);
      const groupMessages = await generateGroupMessages(group.id, 15);
      setMessages(groupMessages);
    } catch (error) {
      console.error('Error loading messages:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSendMessage = () => {
    if (!messageText.trim() || !group) return;
    
    // In a real app, this would send the message to Firebase
    // For now, just add it to the local state
    const newMessage: GroupMessage = {
      id: `new_${Date.now()}`,
      groupId: group.id,
      senderId: '00001', // Current user ID
      senderName: 'You', // We'd get this from the auth context
      content: messageText.trim(),
      timestamp: {
        toDate: () => new Date(),
        seconds: Math.floor(Date.now() / 1000),
        nanoseconds: 0
      } as any,
      readBy: ['00001'] // Current user has read it
    };
    
    setMessages([newMessage, ...messages]);
    setMessageText('');
  };

  const formatMessageTime = (timestamp: any): string => {
    const messageDate = timestamp.toDate();
    const hours = messageDate.getHours();
    const minutes = messageDate.getMinutes();
    return `${hours % 12 || 12}:${minutes.toString().padStart(2, '0')} ${hours >= 12 ? 'PM' : 'AM'}`;
  };

  const renderMessage = ({ item }: { item: GroupMessage }) => {
    const isCurrentUser = item.senderId === '00001';
    
    return (
      <View style={[
        styles.messageContainer,
        isCurrentUser ? styles.currentUserMessage : styles.otherUserMessage
      ]}>
        {!isCurrentUser && (
          <View style={styles.messageSenderInfo}>
            <Text style={styles.messageSenderName}>{item.senderName}</Text>
          </View>
        )}
        
        <View style={[
          styles.messageBubble,
          isCurrentUser ? styles.currentUserBubble : styles.otherUserBubble
        ]}>
          <Text style={[
            styles.messageText,
            isCurrentUser ? styles.currentUserText : styles.otherUserText
          ]}>
            {item.content}
          </Text>
          <Text style={[
            styles.messageTime,
            isCurrentUser ? styles.currentUserTime : styles.otherUserTime
          ]}>
            {formatMessageTime(item.timestamp)}
          </Text>
        </View>
      </View>
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
          
          <View style={styles.groupInfo}>
            <Text style={styles.groupName}>{group.name}</Text>
            <TouchableOpacity 
              onPress={() => onViewMembers(group)}
              style={styles.membersButton}
            >
              <Text style={styles.membersText}>
                {group.memberCount} members
              </Text>
              <Ionicons name="chevron-forward" size={16} color="#666" />
            </TouchableOpacity>
          </View>
          
          <Image 
            source={{ uri: group.groupImageUrl }} 
            style={styles.groupImage}
            defaultSource={{ uri: 'https://via.placeholder.com/100/4a90e2/ffffff?text=Group' }}
          />
        </View>
        
        {loading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#3498db" />
            <Text style={styles.loadingText}>Loading messages...</Text>
          </View>
        ) : (
          <KeyboardAvoidingView
            style={styles.keyboardAvoidingView}
            behavior={Platform.OS === 'ios' ? 'padding' : undefined}
            keyboardVerticalOffset={Platform.OS === 'ios' ? 100 : 0}
          >
            <FlatList
              data={messages}
              renderItem={renderMessage}
              keyExtractor={(item) => item.id}
              contentContainerStyle={styles.messagesList}
              inverted={true} // Latest messages at the bottom
            />
            
            <View style={styles.inputContainer}>
              <TextInput
                style={styles.input}
                placeholder="Type a message..."
                value={messageText}
                onChangeText={setMessageText}
                multiline
              />
              <TouchableOpacity 
                style={[
                  styles.sendButton,
                  !messageText.trim() && styles.sendButtonDisabled
                ]}
                onPress={handleSendMessage}
                disabled={!messageText.trim()}
              >
                <Ionicons name="send" size={24} color="white" />
              </TouchableOpacity>
            </View>
          </KeyboardAvoidingView>
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
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  backButton: {
    marginRight: 16,
  },
  groupInfo: {
    flex: 1,
  },
  groupName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  membersButton: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  membersText: {
    fontSize: 14,
    color: '#666',
    marginRight: 4,
  },
  groupImage: {
    width: 40,
    height: 40,
    borderRadius: 20,
  },
  keyboardAvoidingView: {
    flex: 1,
  },
  messagesList: {
    paddingHorizontal: 16,
    paddingBottom: 16,
  },
  messageContainer: {
    marginVertical: 8,
    maxWidth: '80%',
  },
  currentUserMessage: {
    alignSelf: 'flex-end',
  },
  otherUserMessage: {
    alignSelf: 'flex-start',
  },
  messageSenderInfo: {
    marginBottom: 4,
  },
  messageSenderName: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
  },
  messageBubble: {
    borderRadius: 18,
    padding: 12,
    minWidth: 80,
  },
  currentUserBubble: {
    backgroundColor: '#3498db',
  },
  otherUserBubble: {
    backgroundColor: '#f0f0f0',
  },
  messageText: {
    fontSize: 16,
    marginBottom: 4,
  },
  currentUserText: {
    color: 'white',
  },
  otherUserText: {
    color: '#333',
  },
  messageTime: {
    fontSize: 12,
    alignSelf: 'flex-end',
  },
  currentUserTime: {
    color: 'rgba(255, 255, 255, 0.8)',
  },
  otherUserTime: {
    color: '#999',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
  },
  input: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 10,
    maxHeight: 100,
    marginRight: 12,
  },
  sendButton: {
    backgroundColor: '#3498db',
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendButtonDisabled: {
    backgroundColor: '#ccc',
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
});

export default GroupChatView; 