import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Image,
  SafeAreaView
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS } from '../../utils/theme';

// Sample messages for group chat
const sampleGroupMessages = [
  {
    id: '1',
    text: 'Has anyone toured the units on Michigan Ave?',
    timestamp: '10:30 AM',
    isUser: false,
    sender: 'Michael',
  },
  {
    id: '2',
    text: 'Yes, I went yesterday. The 2BR units are really nice but a bit pricey.',
    timestamp: '10:35 AM',
    isUser: false,
    sender: 'Sarah',
  },
  {
    id: '3',
    text: 'What was the price range?',
    timestamp: '10:37 AM',
    isUser: true,
    sender: 'You',
  },
  {
    id: '4',
    text: 'Around $2,500-3,000 for the 2BR units.',
    timestamp: '10:40 AM',
    isUser: false,
    sender: 'Sarah',
  },
  {
    id: '5',
    text: 'That\'s actually not bad for that location! I was expecting worse.',
    timestamp: '10:42 AM',
    isUser: false,
    sender: 'Michael',
  },
  {
    id: '6',
    text: 'I\'m interested in splitting a unit. Anyone looking for a roommate?',
    timestamp: '10:45 AM',
    isUser: true,
    sender: 'You',
  },
];

const GroupChatDetail = ({ route, navigation }) => {
  // Safely access the group object with default values if undefined
  const group = route.params?.group || {
    id: '0',
    name: 'Group Chat',
    members: 0,
    lastMessage: '',
    time: ''
  };
  
  const [messages, setMessages] = useState(sampleGroupMessages);
  const [inputText, setInputText] = useState('');
  const flatListRef = useRef(null);

  const sendMessage = () => {
    if (inputText.trim() === '') return;
    
    const newMessage = {
      id: Date.now().toString(),
      text: inputText.trim(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isUser: true,
      sender: 'You',
    };
    
    setMessages([...messages, newMessage]);
    setInputText('');
  };

  const renderMessage = ({ item }) => (
    <View style={[
      styles.messageBubble,
      item.isUser ? styles.userMessage : styles.partnerMessage
    ]}>
      {!item.isUser && <Text style={styles.messageSender}>{item.sender}</Text>}
      <Text style={[styles.messageText, item.isUser ? styles.userMessageText : styles.partnerMessageText]}>
        {item.text}
      </Text>
      <Text style={[styles.messageTimestamp, item.isUser ? styles.userTimestamp : styles.partnerTimestamp]}>
        {item.timestamp}
      </Text>
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      {/* Chat Header */}
      <View style={styles.header}>
        <TouchableOpacity 
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        >
          <Ionicons name="chevron-back" size={28} color={COLORS.text} />
        </TouchableOpacity>
        
        <View style={styles.headerProfile}>
          <View style={styles.groupIconContainer}>
            <Ionicons name="people" size={24} color="#fff" />
          </View>
          <View style={styles.profileInfo}>
            <Text style={styles.profileName}>{group.name}</Text>
            <Text style={styles.profileStatus}>{group.members} members</Text>
          </View>
        </View>
        
        <TouchableOpacity style={styles.optionsButton} onPress={() => {}}>
          <Ionicons name="ellipsis-vertical" size={24} color={COLORS.text} />
        </TouchableOpacity>
      </View>
      
      {/* Messages */}
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.messageList}
        onContentSizeChange={() => flatListRef.current.scrollToEnd({ animated: true })}
        onLayout={() => flatListRef.current.scrollToEnd({ animated: true })}
      />
      
      {/* Input Area */}
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
      >
        <View style={styles.inputContainer}>
          <TouchableOpacity style={styles.attachButton}>
            <Ionicons name="add-circle-outline" size={24} color={COLORS.primary} />
          </TouchableOpacity>
          
          <TextInput
            style={styles.textInput}
            placeholder="Type a message..."
            value={inputText}
            onChangeText={setInputText}
            multiline
            maxLength={500}
          />
          
          <TouchableOpacity 
            style={[
              styles.sendButton,
              inputText.trim() === '' ? styles.sendButtonDisabled : {}
            ]}
            onPress={sendMessage}
            disabled={inputText.trim() === ''}
          >
            <Ionicons name="send" size={20} color={inputText.trim() === '' ? '#999' : 'white'} />
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#eaeaea',
  },
  backButton: {
    padding: 5,
  },
  headerProfile: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    marginLeft: 10,
  },
  groupIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: COLORS.primary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  profileInfo: {
    marginLeft: 10,
  },
  profileName: {
    fontWeight: '600',
    fontSize: 16,
    color: COLORS.text,
  },
  profileStatus: {
    fontSize: 12,
    color: '#757575',
  },
  optionsButton: {
    padding: 5,
  },
  messageList: {
    padding: 15,
  },
  messageBubble: {
    maxWidth: '75%',
    paddingHorizontal: 15,
    paddingVertical: 10,
    borderRadius: 18,
    marginBottom: 10,
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: COLORS.primary,
    borderBottomRightRadius: 4,
  },
  partnerMessage: {
    alignSelf: 'flex-start',
    backgroundColor: 'white',
    borderBottomLeftRadius: 4,
  },
  messageSender: {
    fontSize: 12,
    fontWeight: '600',
    color: '#424242',
    marginBottom: 3,
  },
  messageText: {
    fontSize: 16,
  },
  userMessageText: {
    color: 'white',
  },
  partnerMessageText: {
    color: COLORS.text,
  },
  messageTimestamp: {
    fontSize: 11,
    alignSelf: 'flex-end',
    marginTop: 5,
  },
  userTimestamp: {
    color: 'rgba(255,255,255,0.7)',
  },
  partnerTimestamp: {
    color: '#999',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    backgroundColor: 'white',
    borderTopWidth: 1,
    borderTopColor: '#eaeaea',
  },
  attachButton: {
    padding: 5,
  },
  textInput: {
    flex: 1,
    backgroundColor: '#f1f1f1',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 10,
    marginHorizontal: 10,
    fontSize: 16,
    maxHeight: 100,
  },
  sendButton: {
    backgroundColor: COLORS.primary,
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendButtonDisabled: {
    backgroundColor: '#e0e0e0',
  },
});

export default GroupChatDetail; 